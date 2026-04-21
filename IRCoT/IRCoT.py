from typing import TypedDict, List
from pathlib import Path
import sys
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
import json
from datetime import datetime
import re
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import load_vector_store, get_embeddings, get_llm_openai
from prompts import generate_new_query_prompt, generate_final_answer_prompt, judge_answer_prompt

class RAGState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_docs: List[List[str]]
    answer: List[str]
    iteration: int
    max_iterations: int
    is_sufficient: bool
    reason_for_judgment: List[str]
    db_path: str
    top_k: int


def load_retriever(vector_db_path):
    vector_db = load_vector_store(vector_db_path, get_embeddings())
    if vector_db is None:
        raise ValueError(f"Could not load vector store from {vector_db_path}")
    return vector_db


def retrieve_node(state: RAGState):
    # Use last sub-query if available, otherwise fall back to original query
    query = state["sub_queries"][-1] if state.get("sub_queries") else state["query"]
    vector_db = load_retriever(state["db_path"])

    results = vector_db.similarity_search_with_score(query, k=state["top_k"])
    docs = [
        f"Doc: {doc.page_content} (similarity score: {score:.4f})"
        for doc, score in results
    ]

    return {"retrieved_docs": state.get("retrieved_docs", []) + [docs]}


def generate_node(state: RAGState):
    context = "\n\n".join(doc for docs in state["retrieved_docs"] for doc in docs)
    current_answer = state["answer"][-1] if state.get("answer") else ""

    chain = generate_final_answer_prompt | get_llm_openai() | JsonOutputParser()
    parsed = chain.invoke({
        "query": state["query"],
        "answer": current_answer,
        "context": context,
    })

    return {"answer": state.get("answer", []) + [parsed["answer"]]}

def judge_node(state: RAGState):
    chain = judge_answer_prompt | get_llm_openai() | JsonOutputParser()
    parsed = chain.invoke({
        "query": state["query"],
        "answer": state["answer"][-1],
    })

    print(f"Judge response: {parsed}")

    return {
        "is_sufficient": parsed["is_sufficient"],
        "iteration": state["iteration"] + 1,
        "reason_for_judgment": state.get("reason_for_judgment", []) + [parsed["reason"]],
    }


def refine_query_node(state: RAGState):
    context = "\n\n".join(doc for docs in state["retrieved_docs"] for doc in docs)
    current_answer = state["answer"][-1] if state.get("answer") else ""
    sub_queries = state.get("sub_queries", [])

    chain = generate_new_query_prompt | get_llm_openai() | JsonOutputParser()
    parsed = chain.invoke({
        "query": state["query"],
        "sub_question": sub_queries[-1] if sub_queries else "",
        "answer": current_answer,
        "context": context,
    })
    return {"sub_queries": sub_queries + [parsed["query"]]}


def should_continue(state: RAGState):
    if state["is_sufficient"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "refine"


def graph_builder():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("judge", judge_node)
    builder.add_node("refine_query", refine_query_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "judge")
    builder.add_conditional_edges("judge", should_continue, {"refine": "refine_query", "end": END})
    builder.add_edge("refine_query", "retrieve")

    return builder.compile()


if __name__ == "__main__":
    graph = graph_builder()
    result = graph.invoke({
        "query": "What is the difference between the MOMA dataset and the MOMA-LRG dataset?",
        "iteration": 0,
        "max_iterations": 5,
        "retrieved_docs": [],
        "answer": [],
        "sub_queries": [],
        "is_sufficient": False,
        "reason_for_judgment": [],
        "db_path": "outputs/test3/",
        "top_k": 3,
    })

    print(result["answer"][-1])

    log = {
        "timestamp": datetime.now().isoformat(),
        "query": result["query"],
        "answer": result["answer"],
        "final_answer": result["answer"][-1],
        "iterations": result["iteration"],
        "sub_queries": result["sub_queries"],
        "reason_for_judgment": result["reason_for_judgment"],
        "retrieved_docs": result["retrieved_docs"],
        "all_answers": result["answer"],
    }

    log_path = f"logs/langraph/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)