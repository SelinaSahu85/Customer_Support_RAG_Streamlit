from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# ---- API KEY ----
os.environ["GOOGLE_API_KEY"] = "AIzaSyCc8AoPXEw92mQ0D_udmnMW9H-TQiD_odc"

# ---- Load Embeddings ----
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLm-L6-v2"
)

# ---- Load FAISS ----
vectorstore = FAISS.load_local(
    "faiss_support_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# ---- Retriever ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0
)

# ---- Prompt ----
prompt_template = """
You are an enterprise customer support AI assistant.

Use retrieved context to answer the customer query.

Return output in this format:
Issue Category:
Sentiment:
Suggested Resolution:
Escalation Status:

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ---- RAG Chain ----
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# ---- RAG FUNCTION ----
def run_rag(query: str):
    result = qa_chain.invoke(query)
    return result["result"]

print("✅ RAG pipeline ready")