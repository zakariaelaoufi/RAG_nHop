from rounting import determine_routing
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Optional
import argparse
import json
from graph import BinaryTree, Node, Edge
from prompt import rag_query_decomposition_prompt, rag_query_decomposition_tree_prompt
from collections import defaultdict, deque
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from visualize import render_tree_png
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

        # print(f"\n[{node.node_id}] {node.question_placeholder}")
        # print(f"  Answer: {node.answer[:120]}...")

    execute_node(tree.root)
    return tree


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


def run_pipeline(query: str, vector_db_path: str, top_k: int) -> None:
    vector_db = load_vector_store(vector_db_path, get_embeddings())

    if vector_db is None:
        raise ValueError(f"Could not load vector store from {vector_db_path}")

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-5-nano", temperature=0.0),
        retriever=retriever,
        return_source_documents=True
    )

    decision = determine_routing(query)

    if decision.route == "SINGLE_HOP":
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "")
        retrieved_content = result.get("source_documents", [])
    else:
        decomposition = decompose_query(query, rag_query_decomposition_tree_prompt)
        tree = parse_decomposition_to_tree(decomposition)
        execute_rag_tree(tree, qa_chain)
        answer = tree.root.answer
        retrieved_content = tree.root.retrieved_content
        render_tree_png(tree, f"decompositions/test/decomposition_{_safe_filename_fragment(query)}_{uuid.uuid4()}.png", title=query)

    return answer, retrieved_content, decision


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
        render_tree_png(tree, f"decompositions/answered/decomposition_{query_tag}__answered_{uuid.uuid4()}.png")
        print("=" * 50)
        print(f"Final Answer: {tree.root.answer}")
        print("\nSources for root node:")
        for doc in tree.root.retrieved_content:
            print(f"- {doc.page_content[:200]}... (score: {doc.metadata.get('score', 'N/A')})")


if __name__ == "__main__":
    main()