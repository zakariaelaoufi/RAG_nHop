from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Node:
	node_id: str
	question_placeholder: str
	retrieved_content: str
	answer: str
	right: Optional[Node] = None
	left: Optional[Node] = None


@dataclass
class Edge:
	source: str
	target: str
	operation_or_purpose: str

class BinaryTree:
    def __init__(self) -> None:
        self.root: Optional[Node] = None
        self._nodes: dict[str, Node] = {}

    def add_node(
        self,
        node_id: str,
        question_placeholder: str,
        retrieved_content: str = "",
        answer: str = "",
    ) -> Node:
        node = Node(
            node_id=node_id,
            question_placeholder=question_placeholder,
            retrieved_content=retrieved_content,
            answer=answer,
        )
        self._nodes[node_id] = node
        return node

    def get_node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return self._nodes[node_id]

    def link(self, parent_id: str, left_id: Optional[str], right_id: Optional[str]) -> None:
        """Wire left/right children onto a parent node."""
        parent = self.get_node(parent_id)
        if left_id is not None:
            parent.left = self.get_node(left_id)
        if right_id is not None:
            parent.right = self.get_node(right_id)

    def set_root(self, node_id: str) -> None:
        self.root = self.get_node(node_id)


class Graph:
	def __init__(self) -> None:
		self.nodes: Dict[str, Node] = {}
		self.edges: List[Edge] = []

	def add_node(
		self,
		node_id: str,
		question_placeholder: str,
		retrieved_content: str,
		answer: str,
		right: Optional[Node] = None,
		left: Optional[Node] = None,
	) -> None:
		self.nodes[node_id] = Node(
			node_id=node_id,
			question_placeholder=question_placeholder,
			retrieved_content=retrieved_content,
			answer=answer,
			right=right,
			left=left,
		)

	def add_edge(self, source: str, target: str, operation_or_purpose: str) -> None:
		if source not in self.nodes:
			raise ValueError(f"Source node '{source}' does not exist.")
		if target not in self.nodes:
			raise ValueError(f"Target node '{target}' does not exist.")

		self.edges.append(
			Edge(
				source=source,
				target=target,
				operation_or_purpose=operation_or_purpose,
			)
		)

	def get_node(self, node_id: str) -> Node:
		return self.nodes[node_id]
