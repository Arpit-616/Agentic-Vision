# Agentic Vision :-  A Multi Utility Chatbot

A Streamlit-based chatbot that combines:

- PDF question answering with retrieval-augmented generation
- Web search
- Stock price lookup
- Basic calculator support
- Multiple chat threads in the sidebar with friendly labels and context previews

The app uses LangGraph for tool routing, LangChain integrations for models and retrieval, and FAISS for vector search over uploaded PDFs.

## Features

- Upload a PDF and ask questions about its contents
- Keep document context isolated per chat thread
- Start new chats and revisit earlier threads from the sidebar
- Use a built-in calculator tool for simple arithmetic
- Search the web with DuckDuckGo
- Fetch stock quotes with Alpha Vantage
- Run with Ollama

## How It Works

The UI lives in `frontend.py` and the backend logic lives in `backend.py`.

- `frontend.py`
  - Builds the Streamlit interface
  - Tracks active and previous chat threads in `st.session_state`
  - Streams assistant responses in the chat UI
  - Lets users upload PDFs for the current thread

- `backend.py`
  - Loads the configured LLM and embedding model
  - Splits uploaded PDFs into chunks
  - Stores embeddings in a FAISS vector store
  - Exposes tools for search, stock prices, math, and PDF retrieval
  - Runs a LangGraph workflow that decides when to call tools

## Project Structure

```text
.
|-- backend.py
|-- frontend.py
|-- requirements.txt
|-- .env
|-- chatbot.db
```

## Requirements

- Python 3.10+
- `pip`
- Ollama installed locally if using Ollama

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create or update `.env` in the project root.

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Running the App

Start Streamlit with:

```powershell
streamlit run frontend.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Using the App

1. Launch the app.
2. Upload a PDF in the sidebar if you want document-based answers.
3. Ask questions in the chat input.
4. Use `New Chat` to start a separate thread.
5. Open earlier threads from the sidebar.

## Supported Tools

### PDF Retrieval

- Upload a PDF for the active thread
- The file is chunked and embedded
- The assistant uses retrieval to answer questions about that document

Note: the latest PDF uploaded in a thread becomes the active retrieval source for that thread.

### Calculator

The assistant can handle basic operations:

- `add`
- `sub`
- `mul`
- `div`

### Web Search

DuckDuckGo search is available when the assistant decides it needs fresh web information.

### Stock Prices

The app includes a stock lookup tool backed by Alpha Vantage.

## Notes and Limitations

- Chat thread history in the sidebar is currently stored in Streamlit session state.
- PDF retrievers are stored in memory while the app is running.
- If the app restarts, in-memory thread state and PDF indexes are cleared.
- The stock tool may be rate-limited depending on Alpha Vantage usage.
- The current UI hides raw thread UUIDs and shows friendly thread labels instead.

## Troubleshooting

### Ollama model not found

If the configured Ollama model is missing, pull it locally:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Then restart the app.

### No document indexed

If the assistant says no document is available:

- Upload a PDF in the current thread
- Ask the question again after indexing finishes

## Tech Stack

- Streamlit
- LangChain
- LangGraph
- FAISS
- Ollama
- PyPDF
- DuckDuckGo Search
- Requests

## Future Improvements

- Persist chat threads across app restarts
- Persist vector indexes per thread
- Replace the hardcoded stock API key with environment-based configuration
- Add tests and better error handling
