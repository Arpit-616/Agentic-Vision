from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.feature_extraction.text import HashingVectorizer
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)



# -------------------
# 1. LLM + embeddings
# -------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=groq_api_key,
)

class _HashingEmbeddings:
    def __init__(self, n_features: int = 384):
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def _embed(self, text_value: str) -> list[float]:
        return self._vectorizer.transform([text_value]).toarray()[0].astype(float).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text_value) for text_value in texts]

    def embed_query(self, text_value: str) -> list[float]:
        return self._embed(text_value)


_EMBEDDINGS: Optional[_HashingEmbeddings] = None
_DB_ENGINE: Optional[Engine] = None


def _get_embeddings() -> _HashingEmbeddings:
    """Initialize embeddings only when PDF ingestion is used."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = _HashingEmbeddings()
    return _EMBEDDINGS


def _normalize_database_url(database_url: Optional[str]) -> Optional[str]:
    if not database_url:
        return None
    if database_url.startswith("mysql://"):
        return "mysql+pymysql://" + database_url[len("mysql://") :]
    return database_url


def _get_db_engine() -> Optional[Engine]:
    global _DB_ENGINE
    database_url = _normalize_database_url(os.getenv("DATABASE_URL"))
    if not database_url:
        return None
    if _DB_ENGINE is None:
        _DB_ENGINE = create_engine(database_url, pool_pre_ping=True, future=True)
        _ensure_db_schema(_DB_ENGINE)
    return _DB_ENGINE


def _ensure_db_schema(engine: Engine) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id VARCHAR(64) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            thread_id VARCHAR(64) NOT NULL,
            role VARCHAR(32) NOT NULL,
            content LONGTEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_chat_messages_thread_id (thread_id, id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_documents (
            thread_id VARCHAR(64) NOT NULL,
            filename VARCHAR(255) NOT NULL,
            documents INT NOT NULL,
            chunks INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (thread_id, filename)
        )
        """,
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _touch_thread(thread_id: str) -> None:
    engine = _get_db_engine()
    if engine is None:
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO chat_threads (thread_id)
                VALUES (:thread_id)
                ON DUPLICATE KEY UPDATE thread_id = thread_id
                """
            ),
            {"thread_id": str(thread_id)},
        )


def save_thread_message(thread_id: str, role: str, content: str) -> None:
    engine = _get_db_engine()
    if engine is None:
        return
    _touch_thread(thread_id)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO chat_messages (thread_id, role, content)
                VALUES (:thread_id, :role, :content)
                """
            ),
            {
                "thread_id": str(thread_id),
                "role": role,
                "content": content,
            },
        )


def load_thread_messages(thread_id: str) -> list[dict]:
    engine = _get_db_engine()
    if engine is None:
        return []
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT role, content
                FROM chat_messages
                WHERE thread_id = :thread_id
                ORDER BY id ASC
                """
            ),
            {"thread_id": str(thread_id)},
        ).mappings()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def load_thread_ids() -> list[str]:
    engine = _get_db_engine()
    if engine is None:
        return []
    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT thread_id FROM chat_threads ORDER BY created_at ASC")
        ).mappings()
        return [row["thread_id"] for row in rows]


def save_thread_document_metadata(thread_id: str, filename: str, documents: int, chunks: int) -> None:
    engine = _get_db_engine()
    if engine is None:
        return
    _touch_thread(thread_id)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO chat_documents (thread_id, filename, documents, chunks)
                VALUES (:thread_id, :filename, :documents, :chunks)
                ON DUPLICATE KEY UPDATE
                    documents = VALUES(documents),
                    chunks = VALUES(chunks)
                """
            ),
            {
                "thread_id": str(thread_id),
                "filename": filename,
                "documents": int(documents),
                "chunks": int(chunks),
            },
        )


def load_thread_document_metadata(thread_id: str) -> dict:
    engine = _get_db_engine()
    if engine is None:
        return {}
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT filename, documents, chunks
                FROM chat_documents
                WHERE thread_id = :thread_id
                ORDER BY created_at ASC
                """
            ),
            {"thread_id": str(thread_id)},
        ).mappings()
        result: dict = {}
        for row in rows:
            result[row["filename"]] = {
                "filename": row["filename"],
                "documents": row["documents"],
                "chunks": row["chunks"],
            }
        return result

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, _get_embeddings())
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        save_thread_document_metadata(
            thread_id=str(thread_id),
            filename=filename or os.path.basename(temp_path),
            documents=len(docs),
            chunks=len(chunks),
        )

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
_search_runner = DuckDuckGoSearchRun(region="us-en")


@tool
def search_web(query: str) -> dict:
    """Search the web using DuckDuckGo, with network-safe error handling."""
    try:
        return {"query": query, "result": _search_runner.invoke(query)}
    except Exception as exc:
        return {
            "query": query,
            "error": "Web search is unavailable right now.",
            "details": str(exc),
        }


@tool
def brave_search(query: str) -> dict:
    """Compatibility alias for web search tool calls that still use the old name."""
    return search_web(query)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {
            "symbol": symbol,
            "error": "Stock lookup failed.",
            "details": str(exc),
        }


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_web, brave_search, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. Answer directly unless you truly need a tool. "
            "Use `rag_tool` for questions about the uploaded PDF and include the thread_id "
            f"`{thread_id}`. Use the web search, stock price, and calculator tools only when "
            "they are clearly necessary. If document is available, use that document to answer questions instead of web search. If no document is indexed, inform the user that they can upload"
            "a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    try:
        response = llm_with_tools.invoke(messages, config=config)
    except Exception:
        response = llm.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

# -------------------
# 7. Helpers
# -------------------
def thread_document_metadata(thread_id: str) -> dict:
    thread_key = str(thread_id)
    if thread_key in _THREAD_METADATA:
        return _THREAD_METADATA[thread_key]
    return load_thread_document_metadata(thread_key)
