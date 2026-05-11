This project is a production-level AI Customer Support Assistant built using:

✅ Retrieval-Augmented Generation (RAG)
✅ Multi-Agent Architecture (LangGraph)
✅ FastAPI Backend
✅ Streamlit Chat UI
✅ FAISS Vector Search

The system can:

Understand customer queries
Classify issue type
Analyze sentiment
Retrieve relevant context
Generate response
Decide escalation automatically


🧠 Architecture
User → Streamlit App
            ↓
        FastAPI (/query)
            ↓
     Multi-Agent Pipeline
            ↓
  ┌───────────────────────┐
  │ Query Agent           │
  │ Sentiment Agent       │
  │ Retrieval Agent       │
  │ Escalation Agent      │
  │ Response Generator    │
  └───────────────────────┘
            ↓
        Final Answer


📂 Project Structure
CustomerSupport_RAG_Agents_StreamlitApp/
│
├── data/
│   └── customer_support_tickets.csv
│
├── faiss_support_index/
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py             # Data preprocessing + FAISS creation
├── rag_pipeline.py       # Retriever + Gemini LLM setup
├── agents.py             # Multi-agent workflow (LangGraph)
├── app.py                # FastAPI backend
├── streamlit_app.py      # Streamlit frontend UI
├── requirements.txt
└── README.md

