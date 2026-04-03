
# 🤖 Agentic Chat

### A Multi-Utility AI Chatbot (RAG + Tools + Multi-Threading)

Agentic Chat is a **Streamlit-based intelligent chatbot** that combines document understanding, real-time web search, stock lookup, and utility tools into a single seamless interface.

It leverages **LangGraph for agent orchestration**, **LangChain for integrations**, and **FAISS for fast document retrieval**, enabling a powerful Retrieval-Augmented Generation (RAG) system.

---

## 🚀 Live Demo

👉 https://agentic-vision-v30k.onrender.com


---

## ✨ Features

* 📄 **PDF Question Answering (RAG)**

  * Upload PDFs and ask contextual questions
  * Uses embeddings + FAISS vector search

* 🌐 **Web Search**

  * Fetch real-time information via DuckDuckGo

* 📈 **Stock Price Lookup**

  * Get stock data using Alpha Vantage API

* 🧮 **Built-in Calculator**

  * Supports basic arithmetic (add, sub, mul, div)

* 💬 **Multi-Chat Threads**

  * Sidebar-based chat history
  * Friendly labels + context previews

* ⚡ **Streaming Responses**

  * Real-time response generation in UI

---

## 🧠 How It Works

### 🔹 Frontend (`frontend.py`)

* Built with Streamlit
* Manages chat UI and session state
* Handles PDF uploads
* Displays chat threads

### 🔹 Backend (`backend.py`)

* Handles tool routing via LangGraph
* Splits PDFs into chunks
* Generates embeddings and stores in FAISS
* Integrates tools:

  * Web search
  * Calculator
  * Stock API
  * PDF retrieval

---

## 🗂️ Project Structure

```bash
.
├── backend.py        # Core logic and tools
├── frontend.py       # Streamlit UI
├── requirements.txt  # Dependencies
├── .env              # Environment variables
├── chatbot.db        # Optional local DB
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/agentic-chat.git
cd agentic-chat
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=mysql+pymysql://username:password@host:3306/database_name
```

> 💡 `DATABASE_URL` is optional (used for persistence)

---

## ▶️ Run the App

```bash
streamlit run frontend.py
```

Open in browser:

```text
http://localhost:8501
```

---

## 🧰 Supported Tools

### 📄 PDF Retrieval

* Upload a document
* Automatically chunked and embedded
* Used for context-aware answering

### 🌐 Web Search

* DuckDuckGo integration
* Used when fresh data is needed

### 📈 Stock Prices

* Powered by Alpha Vantage API
* Real-time stock lookup

### 🧮 Calculator

Supports:

* `add`
* `sub`
* `mul`
* `div`

---

## 🗄️ Persistence (Optional)

* Chat history → MySQL (via `DATABASE_URL`)
* PDF embeddings → In-memory (FAISS)

> ⚠️ Data resets when app restarts unless DB is configured

---

## ⚠️ Limitations

* In-memory PDF storage (not persistent yet)
* API rate limits (Alpha Vantage)
* No authentication system
* Session-based chat history (unless DB used)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM:** Groq
* **Frameworks:** LangChain, LangGraph
* **Vector DB:** FAISS
* **PDF Processing:** PyPDF
* **Embeddings:** Scikit-learn (local hashing)
* **APIs:** DuckDuckGo, Alpha Vantage

---

## 🚀 Future Improvements

* ✅ Persistent vector database (FAISS → cloud DB)
* ✅ User authentication
* ✅ Chat history across sessions
* ✅ Better error handling
* ✅ API key management via environment
* ✅ UI enhancements

---

## 🐛 Troubleshooting

### Missing API Key

Make sure `.env` contains:

```env
GROQ_API_KEY=your_key
```

---

### No Document Found

* Upload a PDF first
* Wait for indexing
* Then ask your question

---

### MySQL Issues

* Ensure correct `DATABASE_URL`
* Use cloud DB instead of localhost for deployment

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Make changes
4. Submit a pull request

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Support

If you found this useful, consider giving it a ⭐ on GitHub!

---


* Create a **cool project banner**
* Or optimize it for a **hackathon submission page** 🚀
