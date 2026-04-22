# visualize.py
from __future__ import annotations
from typing import Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from graph import BinaryTree, Node


# ── layout ────────────────────────────────────────────────────────────────────

def _assign_positions(
    node: Optional[Node],
    pos: dict[str, tuple[float, float]],
    x: float = 0.0,
    y: float = 0.0,
    x_gap: float = 1.0,
    depth: int = 0,
) -> float:
    """
    Recursive in-order layout (Reingold–Tilford style).
    Returns the x-coordinate assigned to `node`.
    """
    if node is None:
        return x

    # Place left subtree first
    x = _assign_positions(node.left, pos, x, y - 1.5, x_gap / 1.6, depth + 1)

    # Place this node
    pos[node.node_id] = (x, y)
    x += x_gap

    # Place right subtree
    x = _assign_positions(node.right, pos, x, y - 1.5, x_gap / 1.6, depth + 1)

    return x


# ── build nx graph from tree ──────────────────────────────────────────────────

def _tree_to_nx(root: Optional[Node]) -> tuple[nx.DiGraph, dict]:
    """Walk the binary tree and build a directed graph + label map."""
    G = nx.DiGraph()
    labels: dict[str, str] = {}
    is_leaf: dict[str, bool] = {}

    def walk(node: Optional[Node]) -> None:
        if node is None:
            return
        G.add_node(node.node_id)

        # Wrap long questions for readability
        words = node.question_placeholder.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 28:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        labels[node.node_id] = "\n".join(lines)
        is_leaf[node.node_id] = node.left is None and node.right is None

        if node.left:
            G.add_edge(node.node_id, node.left.node_id, side="L")
            walk(node.left)
        if node.right:
            G.add_edge(node.node_id, node.right.node_id, side="R")
            walk(node.right)

    walk(root)
    return G, labels, is_leaf


# ── public API ────────────────────────────────────────────────────────────────

def render_tree_png(
    tree: BinaryTree,
    output_path: str = "tree.png",
    title: str = "Query Decomposition Tree",
) -> str:
    """
    Render the binary tree as a PNG and save it to `output_path`.
    Returns the resolved output path.

    Usage:
        from visualize import render_tree_png
        render_tree_png(tree, "outputs/decomposition.png")
    """
    if tree.root is None:
        raise ValueError("Tree has no root — nothing to render.")

    G, labels, is_leaf = _tree_to_nx(tree.root)

    # Compute positions
    pos: dict[str, tuple[float, float]] = {}
    _assign_positions(tree.root, pos)

    # ── figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")

    # ── edges ────────────────────────────────────────────────────────────────
    edge_labels: dict[tuple, str] = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = data.get("side", "")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#9ca3af",
        width=2.0,
        node_size=3000,       # keeps arrow tips away from node centres
        connectionstyle="arc3,rad=0.05",
    )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=9, font_color="#6b7280",
    )

    # ── nodes — colour by type ────────────────────────────────────────────────
    root_id   = tree.root.node_id
    leaf_ids  = [n for n, leaf in is_leaf.items() if leaf]
    inner_ids = [n for n in G.nodes if n != root_id and n not in leaf_ids]

    def draw_nodes(node_list: list[str], color: str, size: int) -> None:
        if node_list:
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list, ax=ax,
                node_color=color, node_size=size,
                linewidths=2, edgecolors="#374151",
            )

    draw_nodes([root_id], "#6366f1", 4200)   # indigo  — root
    draw_nodes(inner_ids,  "#f59e0b", 3800)   # amber   — internal
    draw_nodes(leaf_ids,   "#10b981", 3400)   # emerald — leaves

    # ── labels ───────────────────────────────────────────────────────────────
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7.5, font_color="black", font_weight="bold",
    )

    # node_id badge (small, below each node)
    id_pos = {nid: (x, y - 0.38) for nid, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, id_pos,
        labels={nid: nid for nid in G.nodes},
        ax=ax, font_size=8, font_color="#1f2937",
    )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#6366f1", label="Root"),
        mpatches.Patch(color="#f59e0b", label="Internal node"),
        mpatches.Patch(color="#10b981", label="Leaf (retrieved)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tree PNG saved -> {output_file}")
    return str(output_file)
