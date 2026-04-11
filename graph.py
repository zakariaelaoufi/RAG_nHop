from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Node:
	node_id: str
	question_placeholder: str
	retrieved_content: str
	answer: str


@dataclass
class Edge:
	source: str
	target: str
	operation_or_purpose: str


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
	) -> None:
		self.nodes[node_id] = Node(
			node_id=node_id,
			question_placeholder=question_placeholder,
			retrieved_content=retrieved_content,
			answer=answer,
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
