from rounting import determine_routing
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Optional
import argparse
import json
from graph import Graph, BinaryTree, Node, Edge
from prompt import rag_query_decomposition_prompt, rag_query_decomposition_tree_prompt
from collections import defaultdict, deque
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from visualize import render_tree_png
import os
import uuid
import re

load_dotenv()


def _safe_filename_fragment(text: str, max_len: int = 20) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._-')
    if not sanitized:
        sanitized = "query"
    return sanitized[:max_len]


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


def decompose_query(query: str, prompt: str) -> dict:
    prompt = prompt.format_messages(query=query)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    response = llm.invoke(prompt)
    response_text = response.content

    try:
        decomposition = json.loads(response_text)
        return decomposition
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {response_text}") from e


# Graph construction and execution logic

# def parse_decomposition_to_graph(decomposition: dict) -> Graph:
#     graph = Graph()
        
#     got = decomposition.get("GraphOfThoughts", {})
        
#     vertices = got.get("vertices", [])
#     for vertex in vertices:
#         graph.add_node(
#             node_id=vertex["node_id"],
#             question_placeholder=vertex["question_placeholder"],
#             retrieved_content=vertex.get("retrieved_content", ""),
#             answer=vertex.get("answer", ""),
#         )
        
#     edges = got.get("edges", [])
#     for edge in edges:
#         graph.add_edge(
#             source=edge["source"],
#             target=edge["target"],
#             operation_or_purpose=edge["operation_or_purpose"],
#         )
        
#     return graph

# def bottom_up_order(graph: Graph):
#     in_degree = {node_id: 0 for node_id in graph.nodes}
#     child_to_parents = defaultdict(list)

#     for edge in graph.edges:
#         child_to_parents[edge.source].append(edge.target)
#         in_degree[edge.target] += 1

#     # Start with leaves (in-degree = 0)
#     queue = deque([n for n in graph.nodes if in_degree[n] == 0])
#     order = []

#     while queue:
#         node = queue.popleft()
#         order.append(node)

#         for parent in child_to_parents[node]:
#             in_degree[parent] -= 1
#             if in_degree[parent] == 0:
#                 queue.append(parent)

#     return order

# def execute_rag_graph(graph: Graph, qa_chain):
#     # Build relationships
#     child_to_parents = defaultdict(list)
#     parent_to_children = defaultdict(list)

#     for edge in graph.edges:
#         child_to_parents[edge.source].append(edge.target)
#         parent_to_children[edge.target].append(edge.source)

#     # Find leaves (no dependencies)
#     leaves = [
#         node_id for node_id in graph.nodes
#         if node_id not in parent_to_children
#     ]

#     visited = set()

#     def execute_node(node_id):
#         if node_id in visited:
#             return graph.get_node(node_id).answer

#         node = graph.get_node(node_id)

#         # 1. Resolve children first
#         children = parent_to_children.get(node_id, [])
#         children_answers = {}

#         for child_id in children:
#             children_answers[child_id] = execute_node(child_id)

#         # 2. Generate answer using QA chain
#         answer = qa_chain.invoke({
#             "query": node.question_placeholder,
#         })

#         node.answer = answer.get("result", "")
#         node.retrieved_content = answer.get("source_documents", [])
#         visited.add(node_id)

#         print(f"Executed Node {node_id}: {node.question_placeholder}")
#         print(f"Answer: {node.answer[:100]}...")
#         print(f"Retrieved Content: {node.retrieved_content[:5][:100]}...")

#         return answer

#     # Execute from all leaves upward
#     for leaf in leaves:
#         execute_node(leaf)

#     return graph

# End of graph construction and execution logic

# Tree logic

def parse_decomposition_to_tree(decomposition: dict) -> BinaryTree:
    """
    Build a BinaryTree from the LLM JSON output.
    Nodes are added first, then left/right pointers are wired
    using the node_id references in each node's 'left' and 'right' fields.
    """
    tree = BinaryTree()
    bt = decomposition.get("BinaryTree", {})

    # Pass 1: create all nodes
    for node_data in bt.get("nodes", []):
        tree.add_node(
            node_id=node_data["node_id"],
            question_placeholder=node_data["question_placeholder"],
            retrieved_content=node_data.get("retrieved_content", ""),
            answer=node_data.get("answer", ""),
        )

    # Pass 2: wire left/right pointers
    for node_data in bt.get("nodes", []):
        tree.link(
            parent_id=node_data["node_id"],
            left_id=node_data.get("left"),
            right_id=node_data.get("right"),
        )

    # Set root
    root_id = bt.get("root")
    if root_id:
        tree.set_root(root_id)

    return tree

def postorder(node: Optional[Node]) -> list[Node]:
    """Return nodes in post-order: left → right → parent."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node]


def retrieve_for_node(node: Node, retriever) -> str:
    """
    Retrieve relevant documents for a single node's question.
    Returns the concatenated page content of top-k docs.
    """
    docs = retriever.invoke(node.question_placeholder)
    return "\n\n".join(doc.page_content for doc in docs)


def execute_rag_tree(tree: BinaryTree, qa_chain) -> BinaryTree:
    """
    Execute the RAG pipeline in post-order (leaves first, root last).
    Each internal node receives its children's answers as extra context.
    """
    def execute_node(node: Optional[Node]) -> None:
        if node is None:
            return

        # Post-order: resolve children first
        execute_node(node.left)
        execute_node(node.right)

        # Build context from children answers
        context_parts: list[str] = []
        if node.left and node.left.answer:
            context_parts.append(f"[Left sub-answer]\n{node.left.answer}")
        if node.right and node.right.answer:
            context_parts.append(f"[Right sub-answer]\n{node.right.answer}")

        if context_parts:
            enriched_query = (
                "\n\n".join(context_parts)
                + f"\n\nUsing the above context, answer: {node.question_placeholder}"
            )
        else:
            # Leaf node — pure retrieval
            enriched_query = node.question_placeholder

        result = qa_chain.invoke({"query": enriched_query})
        node.answer = result.get("result", "")
        node.retrieved_content = result.get("source_documents", [])

        print(f"\n[{node.node_id}] {node.question_placeholder}")
        print(f"  Answer: {node.answer[:120]}...")

    execute_node(tree.root)
    return tree

# End of tree construction and execution logic

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument(
        "query",
        nargs="?",
        default="How does GraphVLM relate to the activity graphs proposed as an abstraction on top of the MOMA dataset?",
        help="The query to answer",
    )
    parser.add_argument(
        "--db",
        # default="outputs/test3",
        help="Path to the FAISS vector store directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved documents to pass to the retriever",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vector_db = load_vector_store(args.db, get_embeddings())
    if vector_db is None:
        raise ValueError(f"Could not load vector store from {args.db}")

    retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.top_k}
        )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-5-nano", temperature=0.0),
        retriever=retriever,
        return_source_documents=True
    )

    query = args.query
    
    decision = determine_routing(query)

    print(f"Routing Decision: {decision.route}, "
          f"Confidence: {decision.confidence}, "
          f"Reason: {decision.reason}"
        )

    if decision.route == "SINGLE_HOP":
        print("Routing to SINGLE_HOP retrieval pipeline...")

        result = qa_chain.invoke({"query": query})

        print(f"Answer: {result['result']}")
        print("\nSources:")
        for doc in result.get("source_documents", []):
            print(f"- {doc.page_content[:200]}... (score: {doc.metadata.get('score', 'N/A')})")

        # print("\n---\n")
        # rs = vector_db.similarity_search_with_score(query, k=3)
        # for doc, score in rs:
        #     print(f"Doc: {doc.page_content[:200]}... (score: {score})")

    # else:
    #     print("Routing to MULTI_HOP reasoning pipeline...")
    #     decomposition = decompose_query(query)
    #     graph_decomposition = parse_decomposition_to_graph(decomposition)
    #     # print(f"Decomposed Query: {decomposition}")
    #     # node = graph_decomposition.get_node("N1", {})
    #     # if node:
    #     #     print(node.question_placeholder)

    #     # Iterate over edges
    #     for edge in graph_decomposition.edges:
    #         print(f"{edge.source} -({edge.operation_or_purpose})-> {edge.target}")
    #         print("---")

    #     for edge in graph_decomposition.edges:
    #         source_node = graph_decomposition.get_node(edge.source)
    #         target_node = graph_decomposition.get_node(edge.target)
    #         print(f"{source_node.question_placeholder} -({edge.operation_or_purpose})-> {target_node.question_placeholder}")
    #         print("---")
    #     print("=== Bottom-Up Execution Order ===")

    #     ordd = bottom_up_order(graph_decomposition)
    #     print(f"Bottom-up order of nodes: {ordd}")

    #     final_graph = execute_rag_graph(graph_decomposition, qa_chain)
    #     # final_answer = final_graph.get_node("N3").answer
    #     last_val = next(reversed(final_graph.nodes.values()))



    #     print(f"Final Answer: {last_val}")

    else:
        print("\nRouting to MULTI_HOP binary tree pipeline...")
        decomposition = decompose_query(query, rag_query_decomposition_tree_prompt)
        tree = parse_decomposition_to_tree(decomposition)
        query_tag = _safe_filename_fragment(query)
        render_tree_png(tree, f"decompositions/questions/decomposition_{query_tag}_{uuid.uuid4()}.png", title=query)

        print("\n=== Post-Order Traversal ===")
        postorder_nodes = postorder(tree.root)
        print(f"Post-order traversal of tree nodes: {postorder_nodes}")
        for node in postorder_nodes:
            l = node.left.node_id if node.left else "null"
            r = node.right.node_id if node.right else "null"
            print(f"  [{node.node_id}] left={l}, right={r} | {node.question_placeholder}")

        execute_rag_tree(tree, qa_chain)

        print(f"\n=== Final Answer (root: {tree.root.node_id}) ===")
        print(tree.root.answer)
        render_tree_png(tree, f"decompositions/answered/decomposition_{query_tag}__answered_{uuid.uuid4()}.png")


if __name__ == "__main__":
    main()