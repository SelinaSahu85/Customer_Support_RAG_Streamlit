from fastapi import FastAPI
from pydantic import BaseModel
from agents import run_agents

# ✅ DEFINE APP FIRST
app = FastAPI()

# ✅ Request Schema
class QueryRequest(BaseModel):
    query: str

# ✅ API Endpoint
@app.post("/query")
def query_api(data: QueryRequest):
    result = run_agents(data.query)

    return {
        "query": data.query,
        "category": result["category"],
        "sentiment": result["sentiment"],
        "answer": result["final_response"],
        "escalation": result["escalation"]
    }

