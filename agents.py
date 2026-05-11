from typing import TypedDict
from langgraph.graph import StateGraph, END
from rag_pipeline import retriever, llm

# ---- Shared State ----
class AgentState(TypedDict):
    query: str
    category: str
    sentiment: str
    retrieved_docs: str
    escalation: str
    final_response: str

# ---- Agent 1: Query Classifier ----
def query_agent(state):
    prompt = f"""
    Classify into:
    Login Issue, Billing Issue, Refund Issue, Technical Issue, Account Issue

    Query:
    {state['query']}

    Return ONLY category.
    """
    response = llm.invoke(prompt)
    return {"category": response.content.strip()}

# ---- Agent 2: Sentiment ----
def sentiment_agent(state):
    prompt = f"""
    Sentiment:
    Positive, Neutral, Negative

    Query:
    {state['query']}
    """
    response = llm.invoke(prompt)
    return {"sentiment": response.content.strip()}

# ---- Agent 3: Retrieval ----
def rag_tool(query):
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

def retrieval_agent(state):
    return {"retrieved_docs": rag_tool(state["query"])}

# ---- Agent 4: Escalation ----
def escalation_agent(state):
    if len(state["retrieved_docs"].strip()) == 0:
        return {"escalation": "Escalate to Human Support"}

    if state["sentiment"] == "Negative":
        return {"escalation": "Escalate to Human Support"}

    return {"escalation": "No Escalation Required"}

# ---- Agent 5: Response ----
def response_agent(state):
    prompt = f"""
    Generate short response (<20 words)

    Query: {state['query']}
    Category: {state['category']}
    Sentiment: {state['sentiment']}
    Context: {state['retrieved_docs']}
    """
    response = llm.invoke(prompt)
    return {"final_response": response.content.strip()}

# ---- Workflow ----
workflow = StateGraph(AgentState)

workflow.add_node("query_agent", query_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("escalation_agent", escalation_agent)
workflow.add_node("response_agent", response_agent)

workflow.set_entry_point("query_agent")

workflow.add_edge("query_agent", "sentiment_agent")
workflow.add_edge("sentiment_agent", "retrieval_agent")
workflow.add_edge("retrieval_agent", "escalation_agent")
workflow.add_edge("escalation_agent", "response_agent")
workflow.add_edge("response_agent", END)

app = workflow.compile()

# ---- FINAL FUNCTION ----
def run_agents(query: str):
    result = app.invoke({"query": query})
    return result

print("✅ Agents pipeline ready")