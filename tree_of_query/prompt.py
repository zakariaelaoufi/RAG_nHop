from langchain_core.prompts import ChatPromptTemplate

rag_routing_prompt = ChatPromptTemplate.from_template("""
You are a specialized AI router responsible for classifying user queries 
and routing them to the appropriate Retrieval-Augmented Generation (RAG) pipeline.

Your task is to determine whether the query requires:
1. SINGLE_HOP retrieval
2. MULTI_HOP reasoning

---

### Definitions

SINGLE_HOP:
- The query can be answered using a single piece of information or a single document.
- Requires direct lookup or definition.
- No intermediate reasoning steps are needed.

Examples:
- "What is the capital of France?"
- "What is positional encoding?"
- "Who is Alan Turing?"

---

MULTI_HOP:
- The query requires combining multiple pieces of information.
- Requires reasoning, comparison, or relationships between concepts.
- Involves implicit or explicit multi-step thinking.

Examples:
- "Who is the president of the country that has the Eiffel Tower?"
- "What is the difference between machine learning and deep learning?"
- "How does retrieval-augmented generation improve LLM performance?"
- "What is the relationship between X and Y?"

---

### Decision Rules (IMPORTANT)

- If answering requires **linking multiple facts**, choose MULTI_HOP.
- If the query contains **comparison, relationship, cause-effect, or aggregation**, choose MULTI_HOP.
- If the query is a **simple definition, fact, or entity lookup**, choose SINGLE_HOP.
- If unsure, default to MULTI_HOP (safer for reasoning-heavy queries).

---

### Output Format (STRICT)

Return ONLY a JSON object with no explanation:

{{
	"route": "SINGLE_HOP" | "MULTI_HOP",
	"confidence": float (0 to 1),
	"reason": "short justification (max 15 words)"
}}

---

### Query

{query}
"""
)

rag_query_decomposition_tree_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that decomposes complex queries for a
Retrieval-Augmented Generation (RAG) system using a BINARY TREE structure.

========================================
CORE RULE — BINARY DECOMPOSITION ONLY
========================================
- Every node may have AT MOST 2 children (left, right).
- MAX DEPTH — the tree must not exceed depth 3 (root = depth 0).
- If a sub-query is still complex, decompose it into at most 2 children.
- ATOMIC STOP CONDITION — stop when a question can be answered by a SINGLE document passage with no cross-document synthesis.
- NON-OVERLAPPING CHILDREN — siblings must cover strictly different aspects (no paraphrasing or subsumption).
- NO PARENT PARAPHRASE — children must not reword the parent.
- ASYMMETRY IS ALLOWED — one branch can stop while the other continues.
- BALANCE PREFERENCE — prefer balanced depth when complexity is similar.
- Leaf nodes are directly retrievable atomic questions.
- The root node synthesizes all children answers.

========================================
DECOMPOSITION STRATEGY
========================================
1. Identify the core intent of the query.
2. Split into AT MOST 2 sub-questions.
3. If a sub-question is still complex, recursively decompose it.
4. Stop when the question becomes atomic and directly retrievable.
5. Ensure the root question is the final synthesis query.

========================================
QUERY FORMULATION RULES
========================================
Each `question_placeholder` must:
- Express exactly ONE information need.
- Be atomic at its level (leaves must be directly retrievable).
- Be ≤15 words (max 20).
- Be retrieval-optimized (search-style, not instructional).
- Be phrased as a query (noun phrase or direct question).
- Avoid words like "explain", "describe", "tell me".
- Be unique across nodes.
- Add new informational value (no paraphrasing across nodes).

========================================
NODE STRUCTURE (REQUIRED FIELDS)
========================================
Each node MUST include:
- `node_id`:
    Root          → "N1"
    Left child    → "N1_L"
    Right child   → "N1_R"
    Deeper levels → "N1_L_L", "N1_L_R", etc.
- `question_placeholder`: self-contained retrievable query
- `retrieved_content`: ALWAYS ""
- `answer`: ALWAYS ""
- `left`: node_id or null
- `right`: node_id or null

========================================
STRICT RULES
========================================
- Max 2 children per node.
- Leaves must have `left = null` and `right = null`.
- No cycles.
- NEVER paraphrase the parent question.
- Return VALID JSON ONLY (no markdown, no explanations).

========================================
SCHEMA
========================================
{{
  "BinaryTree": {{
    "user_query": "string",
    "root": "node_id",
    "nodes": [
      {{
        "node_id": "string",
        "question_placeholder": "string",
        "retrieved_content": "",
        "answer": "",
        "left": "node_id or null",
        "right": "node_id or null"
      }}
    ]
  }}
}}

========================================
EXAMPLE 1 — Symmetric Tree
========================================
Query: "Compare the training strategies and downstream performance of Model A and Model B"

{{
  "BinaryTree": {{
    "user_query": "Compare the training strategies and downstream performance of Model A and Model B",
    "root": "N1",
    "nodes": [
      {{
        "node_id": "N1",
        "question_placeholder": "Compare the training strategies and downstream performance of Model A and Model B",
        "retrieval_hint": "n/a",
        "retrieved_content": "",
        "answer": "",
        "left": "N1_L",
        "right": "N1_R"
      }},
      {{
        "node_id": "N1_L",
        "question_placeholder": "What are the training strategy differences between Model A and Model B?",
        "retrieval_hint": "n/a",
        "retrieved_content": "",
        "answer": "",
        "left": "N1_L_L",
        "right": "N1_L_R"
      }},
      {{
        "node_id": "N1_R",
        "question_placeholder": "What are the downstream performance differences between Model A and Model B?",
        "retrieval_hint": "n/a",
        "retrieved_content": "",
        "answer": "",
        "left": "N1_R_L",
        "right": "N1_R_R"
      }},
      {{
        "node_id": "N1_L_L",
        "question_placeholder": "What training objective and data does Model A use?",
        "retrieval_hint": "factual",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }},
      {{
        "node_id": "N1_L_R",
        "question_placeholder": "What training objective and data does Model B use?",
        "retrieval_hint": "factual",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }},
      {{
        "node_id": "N1_R_L",
        "question_placeholder": "Model A benchmark scores on downstream tasks",
        "retrieval_hint": "comparative",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }},
      {{
        "node_id": "N1_R_R",
        "question_placeholder": "Model B benchmark scores on downstream tasks",
        "retrieval_hint": "comparative",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }}
    ]
  }}
}}

========================================
EXAMPLE 2 — Asymmetric Tree
========================================
Query: "What caused the 2008 financial crisis and how did the Dodd-Frank Act respond?"

{{
  "BinaryTree": {{
    "user_query": "What caused the 2008 financial crisis and how did the Dodd-Frank Act respond?",
    "root": "N1",
    "nodes": [
      {{
        "node_id": "N1",
        "question_placeholder": "What caused the 2008 financial crisis and how did the Dodd-Frank Act respond?",
        "retrieval_hint": "n/a",
        "retrieved_content": "",
        "answer": "",
        "left": "N1_L",
        "right": "N1_R"
      }},
      {{
        "node_id": "N1_L",
        "question_placeholder": "What were the primary causes of the 2008 financial crisis?",
        "retrieval_hint": "n/a",
        "retrieved_content": "",
        "answer": "",
        "left": "N1_L_L",
        "right": "N1_L_R"
      }},
      {{
        "node_id": "N1_R",
        "question_placeholder": "Key regulatory provisions introduced by the Dodd-Frank Act",
        "retrieval_hint": "factual",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }},
      {{
        "node_id": "N1_L_L",
        "question_placeholder": "Role of mortgage-backed securities in the 2008 financial crisis",
        "retrieval_hint": "causal",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }},
      {{
        "node_id": "N1_L_R",
        "question_placeholder": "How did regulatory failures contribute to the 2008 financial crisis?",
        "retrieval_hint": "causal",
        "retrieved_content": "",
        "answer": "",
        "left": null,
        "right": null
      }}
    ]
  }}
}}

<Query>{query}</Query>
""")

final_answer_hirarchy_prompt = ChatPromptTemplate.from_template("""
You are given a hierarchical question decomposition tree of a complex query with retrieved evidence for each atomic question.

## Main Question
{main_question}

## Hierarchical decomposition with retrieved evidence
{hierarchy_template}

## Instructions
- Answer the **main question only** using the retrieved evidence provided above.
- Follow the reasoning bottom-up: use the atomic questions (leaves) answers to build up to the final answer.
- Do NOT answer sub-questions individually, do NOT list intermediate steps.
- Your final answer must be a single, coherent, well-structured response.
- Only use information present in the retrieved chunks. Do not hallucinate or add external knowledge.
- If the retrieved evidence is insufficient to answer the main question, say so explicitly.

## Answer
"""
)

final_answer_synthesis_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant responsible for synthesizing a final answer to a complex user query.

The query has been decomposed into a BINARY TREE of sub-queries. Each node in the tree contains:
- a `question_placeholder` (context of the sub-question)
- an `answer` (generated from retrieved content)

Your task is to combine these into a single, coherent, and accurate final answer.

You MUST use ONLY the provided `question_placeholder` and `answer` fields.
Do NOT introduce external knowledge, assumptions, or missing information.
  
---

### Input Format
{{
  "user_query": "string",
  "sub_queries": [
    {{
      "question_placeholder": "string",
      "answer": ["string", ...]
    }}
  ]
}}

---

### Instructions
- Treat each (`question_placeholder`, `answer`) pair as a validated information unit.
- Extract the key facts from each sub-query.
- Combine them logically to answer the original `user_query`.
- Ensure the final answer is:
  - Directly aligned with the user query
  - Coherent and well-structured
  - Concise (no redundancy, no filler)
- Resolve overlaps by merging information, not repeating it.
- If information conflicts, reflect the uncertainty or present both sides clearly.
- If information is incomplete, answer as fully as possible without guessing.

---

### Output Format
Return ONLY a valid JSON object:
{{
  "final_answer": "string"
}}

---

### Example

Input:
{{
  "user_query": "What are the key differences between Model A and Model B in terms of training strategies?",
  "sub_queries": [
    {{
      "question_placeholder": "What is the training strategy of Model A?",
      "answer": ["Model A uses supervised learning with a large labeled dataset."]
    }},
    {{
      "question_placeholder": "What is the training strategy of Model B?",
      "answer": ["Model B uses unsupervised pretraining followed by fine-tuning on a smaller labeled dataset."]
    }}
  ]
}}

Output:
{{
  "final_answer": "Model A relies on supervised learning with a large labeled dataset, whereas Model B uses unsupervised pretraining followed by fine-tuning on a smaller labeled dataset."
}}
"""
)

rag_query_decomposition_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that decomposes complex queries for a Retrieval-Augmented Generation (RAG) system.

Your goal is to:
1. Break the query into the MINIMUM number of essential sub-queries required to answer the question
2. Ensure the subqueries are direct, concise, and retrievable
2. Preserve logical relationships between sub-queries
3. Provide a structure that enables reconstruction of the final answer

STRICT MINIMALITY CONSTRAINT:
- MUST generate ONLY the most critical sub-queries
- Avoid over-decomposition
- Maximum number of nodes: 2–6 (hard limit unless absolutely necessary)
- Each node must contribute directly to the final answer (no auxiliary or redundant queries)

DECOMPOSITION STRATEGY:
- Identify the core intent of the query first
- Extract only indispensable entities, constraints, or comparisons
- Merge closely related sub-questions instead of splitting them
- Prefer broader, information-rich queries over many narrow ones
                                                                  
QUERY FORMULATION (Critical)
Each `question_placeholder` must:
- Express exactly one information need.
- Be atomic (no compound or multi-part questions).
- Be concise (preferably ≤15 words, max 20).
- Be retrieval-optimized (search-style, not instructional).

Output constraints (MUST match runtime graph structure):
- Use exact top-level shape: `{{"GraphOfThoughts": {{...}}}}`
- `GraphOfThoughts.user_query` must be the original user question (verbatim or near-verbatim)

VERTICES:
- `GraphOfThoughts.vertices` maps to Node fields and EVERY node must include:
  - `node_id` (string, unique)
  - `question_placeholder` (self-contained, retrievable, information-dense query)
  - `retrieved_content` (string, ALWAYS "")
  - `answer` (string, ALWAYS "")

EDGES:
- `GraphOfThoughts.edges` maps to Edge fields and EVERY edge must include:
  - `source` (existing node_id)
  - `target` (existing node_id)
  - `operation_or_purpose` (e.g., dependency, comparison, join_thoughts, filter)

STRICT RULES:
- Do NOT use alternative key names (forbidden: source_node, target_node, operation)
- Every edge MUST reference valid node_ids
- Avoid duplicate edges
- Prefer DAG structure (no cycles unless absolutely required)
- Keep graph SMALL but COMPLETE

QUALITY REQUIREMENTS:
- Each sub-query must be:
  * self-contained
  * directly answerable via retrieval
  * necessary for solving the main query
- Ensure relationships between nodes are explicit and meaningful

Return output as VALID JSON ONLY (no markdown, no explanations) and strictly follow this schema:

{{
  "GraphOfThoughts": {{
    "user_query": "string",
    "vertices": [
      {{
        "node_id": "string",
        "question_placeholder": "string",
        "retrieved_content": "",
        "answer": ""
      }}
    ],
    "edges": [
      {{
        "source": "string",
        "target": "string",
        "operation_or_purpose": "string"
      }}
    ]
  }}
}}
                                                                  
example:

{{
  "GraphOfThoughts": {{
    "user_query": "What is the relationship between X and Y?",
    "vertices": [
      {{
        "node_id": "N1",
        "question_placeholder": "What are its key properties of X?",
        "retrieved_content": "",
        "answer": "",
        "right": null,
        "left": null
      }},
      {{
        "node_id": "N2",
        "question_placeholder": "What are its key properties of Y?",
        "retrieved_content": "",
        "answer": "",
        "right": null,
        "left": null
      }},
      {{
        "node_id": "N3",
        "question_placeholder": "What is the relationship between X and Y?",
        "retrieved_content": "",
        "answer": "",
        "right": N1,
        "left": N2
      }}
    ],
    "edges": [
      {{
        "source": "N1",
        "target": "N3",
        "operation_or_purpose": "dependency"
      }},
      {{
        "source": "N2",
        "target": "N3",
        "operation_or_purpose": "dependency"
      }}
    ]
  }}
}}
                                                                  
<Query>{query}</Query>
""")