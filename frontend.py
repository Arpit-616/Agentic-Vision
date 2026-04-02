import uuid
import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    ingest_pdf,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def _shorten_text(text: str, max_chars: int = 80) -> str:
    normalized = " ".join((text or "").split())
    if not normalized:
        return "No context yet."
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def register_thread(thread_id: str) -> str:
    thread_key = str(thread_id)
    st.session_state["thread_messages"].setdefault(thread_key, [])
    if thread_key not in st.session_state["thread_order"]:
        st.session_state["thread_order"].append(thread_key)
    return thread_key


def activate_thread(thread_id: str) -> str:
    thread_key = register_thread(thread_id)
    st.session_state["thread_id"] = thread_key
    st.session_state["message_history"] = list(
        st.session_state["thread_messages"][thread_key]
    )
    return thread_key


def append_message(role: str, content: str):
    message = {"role": role, "content": content}
    st.session_state["message_history"].append(message)

    thread_key = str(st.session_state["thread_id"])
    register_thread(thread_key)
    st.session_state["thread_messages"][thread_key].append(message)


def get_thread_label(thread_id: str) -> str:
    thread_key = str(thread_id)
    order = st.session_state.get("thread_order", [])
    if thread_key in order:
        return f"Thread {order.index(thread_key) + 1}"
    return "Thread"


def get_thread_preview(thread_id: str) -> str:
    thread_key = str(thread_id)
    messages = st.session_state.get("thread_messages", {}).get(thread_key, [])

    for message in messages:
        preview = _shorten_text(message.get("content", ""))
        if preview != "No context yet.":
            speaker = "You" if message.get("role") == "user" else "Assistant"
            return f"{speaker}: {preview}"

    docs = st.session_state.get("ingested_docs", {}).get(thread_key, {})
    if docs:
        latest_doc = list(docs.values())[-1]
        filename = latest_doc.get("filename", "Indexed PDF")
        return _shorten_text(f"PDF: {filename}")

    return "No context yet."


def reset_chat():
    activate_thread(generate_thread_id())


# ======================= Session Initialization ===================
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_messages" not in st.session_state:
    st.session_state["thread_messages"] = {}

if "thread_order" not in st.session_state:
    st.session_state["thread_order"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

thread_key = activate_thread(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Current Thread:** `{get_thread_label(thread_key)}`")
st.sidebar.caption(get_thread_preview(thread_key))

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

previous_threads = [
    previous_thread
    for previous_thread in reversed(st.session_state["thread_order"])
    if previous_thread != thread_key
]
if previous_threads:
    st.sidebar.subheader("Previous Threads")
    for previous_thread in previous_threads:
        with st.sidebar.container():
            st.markdown(f"**{get_thread_label(previous_thread)}**")
            st.caption(get_thread_preview(previous_thread))
            if st.button(
                "Open",
                key=f"switch_thread_{previous_thread}",
                use_container_width=True,
            ):
                activate_thread(previous_thread)
                st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="PDF indexed", state="complete", expanded=False)

# ============================ Main Layout ========================
st.title(" Agentic Vision ")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    append_message("user", user_input)
    with st.chat_message("user"):
        st.text(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    stream_failed = False
    ai_message = ""

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using `{tool_name}` ...",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            stream_failed = True
            error_text = str(exc)
            if "invalid_api_key" in error_text or "Incorrect API key provided" in error_text:
                st.error(
                    "OpenAI authentication failed. Update OPENAI_API_KEY in your .env "
                    "with a real key from https://platform.openai.com/account/api-keys "
                    "and restart Streamlit."
                )
            elif "status code: 404" in error_text and "model" in error_text and "not found" in error_text:
                ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
                st.error(
                    "Ollama model not found locally. Run this in terminal: "
                    f"`ollama pull {ollama_model}` and restart Streamlit."
                )
            else:
                st.error(f"Request failed: {error_text}")

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool finished", state="complete", expanded=False
            )

    if not stream_failed and ai_message:
        append_message("assistant", ai_message)

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()
