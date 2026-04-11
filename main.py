from rounting import determine_routing
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Optional
import json
from graph import Graph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from prompt import rag_query_decomposition_prompt
from dotenv import load_dotenv
import os

load_dotenv()


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def load_vector_store(source_path: str, embeddings) -> Optional[FAISS]:
    try:
        vector_store_path = Path(source_path)
        if vector_store_path.suffix == ".faiss":
            vector_store_path = vector_store_path.parent

        if vector_store_path.exists():
            return FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Vector store path '{vector_store_path}' does not exist.")
            return None
    except Exception as e:
        raise ValueError(f"Failed to load vector store: {e}")


def decompose_query(query: str) -> dict:
    prompt = rag_query_decomposition_prompt.format_messages(query=query)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    response = llm.invoke(prompt)
    response_text = response.content

    try:
        decomposition = json.loads(response_text)
        return decomposition
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {response_text}") from e

def parse_decomposition_to_graph(decomposition: dict) -> Graph:
    graph = Graph()
        
    got = decomposition.get("GraphOfThoughts", {})
        
    vertices = got.get("vertices", [])
    for vertex in vertices:
        graph.add_node(
            node_id=vertex["node_id"],
            question_placeholder=vertex["question_placeholder"],
            retrieved_content=vertex.get("retrieved_content", ""),
            answer=vertex.get("answer", ""),
        )
        
    edges = got.get("edges", [])
    for edge in edges:
        graph.add_edge(
            source=edge["source"],
            target=edge["target"],
            operation_or_purpose=edge["operation_or_purpose"],
        )
        
    return graph

def main():
    vector_db = load_vector_store("outputs/test1", get_embeddings())

    query = "How does the mathematical application of the rotary positional embedding method differ from the baseline positional encoding?"
    decision = determine_routing(query)

    print(f"Routing Decision: {decision.route}, "
          f"Confidence: {decision.confidence}, "
          f"Reason: {decision.reason}"
        )

    if decision.route == "SINGLE_HOP":
        print("Routing to SINGLE_HOP retrieval pipeline...")

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-5-nano", temperature=0.0),
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": query})

        print(f"Answer: {result['result']}")
        print("\nSources:")
        for doc in result.get("source_documents", []):
            print(f"- {doc.page_content[:200]}... (score: {doc.metadata.get('score', 'N/A')})")

        # print("\n---\n")
        # rs = vector_db.similarity_search_with_score(query, k=3)
        # for doc, score in rs:
        #     print(f"Doc: {doc.page_content[:200]}... (score: {score})")

    else:
        print("Routing to MULTI_HOP reasoning pipeline...")
        decomposition = decompose_query(query)
        graph_decomposition = parse_decomposition_to_graph(decomposition)
        # print(f"Decomposed Query: {decomposition}")
        # node = graph_decomposition.get_node("N1", {})
        # if node:
        #     print(node.question_placeholder)

        # Iterate over edges
        for edge in graph_decomposition.edges:
            print(f"{edge.source} -({edge.operation_or_purpose})-> {edge.target}")
            print("---")

        for edge in graph_decomposition.edges:
            source_node = graph_decomposition.get_node(edge.source)
            target_node = graph_decomposition.get_node(edge.target)
            print(f"{source_node.question_placeholder} -({edge.operation_or_purpose})-> {target_node.question_placeholder}")
            print("---")

if __name__ == "__main__":
    main()