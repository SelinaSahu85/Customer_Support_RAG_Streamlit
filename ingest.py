import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ---- Load Dataset ----
df = pd.read_csv("D:/OneDrive - Coforge Limited/Documents/Capstone-2/customer_support_tickets.csv")

# ---- Fill Missing ----
df = df.fillna("Unknown")

# ---- Combine Text ----
def create_text(row):
    return f"""
    Ticket ID: {row['Ticket ID']}
    Subject: {row['Ticket Subject']}
    Description: {row['Ticket Description']}
    Priority: {row['Ticket Priority']}
    Status: {row['Ticket Status']}
    Resolution time: {row['Time to Resolution']}
    Customer Rating: {row['Customer Satisfaction Rating']}
    """

df["combined_text"] = df.apply(create_text, axis=1)

# ---- Convert to Documents ----
documents = [Document(page_content=text) for text in df["combined_text"].tolist()]

# ---- Chunking ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ---- Embeddings ----
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLm-L6-v2"
)

# ---- Vector DB ----
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_support_index")

print("✅ FAISS index created & saved")