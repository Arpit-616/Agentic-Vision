# Agentic Vision :- A Multi Utility Chatbot

A Streamlit-based chatbot that combines:

- PDF question answering with retrieval-augmented generation
- Web search
- Stock price lookup
 - Python 3.10+
 - `pip`
 - A Groq API key
 - Optional MySQL database URL for persistence
- Multiple chat threads in the sidebar with friendly labels and context previews

The app uses LangGraph for tool routing, LangChain integrations for models and retrieval, and FAISS for vector search over uploaded PDFs.

## Features

- Upload a PDF and ask questions about its contents
- Start new chats and revisit earlier threads from the sidebar
- Use a built-in calculator tool for simple arithmetic
- Search the web with DuckDuckGo
- Fetch stock quotes with Alpha Vantage
 ```powershell
 pip install -r requirements.txt
 ```
## How It Works

The UI lives in `frontend.py` and the backend logic lives in `backend.py`.

- `frontend.py`
  - Builds the Streamlit interface
  - Tracks active and previous chat threads in `st.session_state`
 DATABASE_URL=mysql+pymysql://username:password@host:3306/database_name
  - Streams assistant responses in the chat UI
  - Lets users upload PDFs for the current thread

- `backend.py`
 - Set `DATABASE_URL` if you want chat history and PDF metadata stored in MySQL
  - Splits uploaded PDFs into chunks
  - Stores embeddings in a FAISS vector store
  - Exposes tools for search, stock prices, math, and PDF retrieval
 - Chat thread history and PDF metadata can be stored in MySQL when `DATABASE_URL` is set.
 - PDF retrievers are still stored in memory while the app is running.
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

- Optional MySQL database URL for persistence

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
DATABASE_URL=mysql+pymysql://username:password@host:3306/database_name
.\venv\Scripts\activate
```

2. Install dependencies:
- Chat thread history and PDF metadata can also be stored in MySQL when `DATABASE_URL` is set.
- PDF retrievers are still stored in memory while the app is running.
pip install -r requirements.txt
```

## Environment Variables

Create or update `.env` in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
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

### Missing API keys

If the app says an API key is missing:

- Set `GROQ_API_KEY` for the chat model

### No document indexed

If the assistant says no document is available:

- Upload a PDF in the current thread
- Ask the question again after indexing finishes

## Tech Stack

- Streamlit
- LangChain
- LangGraph
- FAISS
- Groq
- PyPDF
- Scikit-learn (local hashing embeddings)
- Requests

## Future Improvements

- Persist chat threads across app restarts
- Persist vector indexes per thread
- Replace the hardcoded stock API key with environment-based configuration
- Add tests and better error handling
