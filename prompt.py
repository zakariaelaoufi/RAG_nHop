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
        "answer": ""
      }},
      {{
        "node_id": "N2",
        "question_placeholder": "What are its key properties of Y?",
        "retrieved_content": "",
        "answer": ""
      }},
      {{
        "node_id": "N3",
        "question_placeholder": "What is the relationship between X and Y?",
        "retrieved_content": "",
        "answer": ""
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