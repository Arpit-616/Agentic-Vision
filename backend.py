from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv(override=True)


def _normalize_ollama_base_url(raw_url: str) -> str:
    """Normalize common invalid local bind addresses to a client-connect URL."""
    cleaned = (raw_url or "").strip().strip('"').strip("'")
    if not cleaned:
        return "http://localhost:11434"

    try:
        parsed = urlparse(cleaned)
    except Exception:
        return "http://localhost:11434"

    if not parsed.scheme or not parsed.netloc:
        return "http://localhost:11434"

    if parsed.hostname in {"0.0.0.0", "::", "[::]"}:
        host = "localhost"
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        else:
            netloc = host
        parsed = parsed._replace(netloc=netloc)

    return urlunparse(parsed).rstrip("/")


def _get_ollama_installed_models(base_url: str) -> list[str]:
    """Fetch model names available in local Ollama; return empty list if unavailable."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(tags_url, timeout=5)
        response.raise_for_status()
        payload = response.json() or {}
        models = payload.get("models", [])
        return [m.get("name", "").strip() for m in models if m.get("name")]
    except Exception:
        return []

# -------------------
# 1. LLM + embeddings
# -------------------
ollama_base_url = _normalize_ollama_base_url(
    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
ollama_model = (
    os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip().strip('"').strip("'")
)
ollama_embed_model = (
    os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    .strip()
    .strip('"')
    .strip("'")
)
installed_models = _get_ollama_installed_models(ollama_base_url)

if installed_models:
    if ollama_model not in installed_models:
        ollama_model = installed_models[0]
    if ollama_embed_model not in installed_models:
        # Fall back to the active chat model to avoid immediate 404 errors.
        ollama_embed_model = ollama_model

llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_base_url,
    temperature=0.2,
)
embeddings = OllamaEmbeddings(model=ollama_embed_model, base_url=ollama_base_url)

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

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

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


tools = [search_web, get_stock_price, calculator, rag_tool]
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
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
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
    return _THREAD_METADATA.get(str(thread_id), {})
