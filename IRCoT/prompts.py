from langchain_core.prompts import ChatPromptTemplate


generate_new_query_prompt = ChatPromptTemplate.from_template("""
You are an information retrieval assistant specialized in query refinement.

Your goal is to generate ONE focused follow-up sub-question that retrieves
missing information required to answer the original query.

### TASK
Given:
- The original question
- The current sub-question
- The current answer (incomplete)
- The retrieved context

You must:
1. Identify the precise information gap in the current answer.
2. Generate ONE new sub-question that directly targets this gap.

### QUERY CONSTRAINTS
The generated sub-question MUST:
- Express exactly ONE information need
- Be atomic and self-contained
- Be ≤ 15 words (hard limit: 20)
- Be retrieval-optimized (search-style, not instructional)
- Be phrased as a query (direct question or noun phrase)
- Avoid vague verbs: "explain", "describe", "tell me"
- Avoid redundancy with previous sub-questions
- Introduce NEW informational value (no paraphrasing)

### INPUT
Original Question:
{query}

Current Sub-question:
{sub_question}

Current Answer:
{answer}

Retrieved Context:
{context}

### OUTPUT FORMAT
Return ONLY a valid JSON object:
{{
  "query": "your generated sub-question"
}}
""")

judge_answer_prompt = ChatPromptTemplate.from_template("""
You are an information retrieval evaluator.

Your task is to determine whether the current answer is SUFFICIENT to fully
and correctly answer the original question.

### DECISION CRITERIA
Mark the answer as sufficient ONLY if:
- It directly and completely answers the original question
- It includes all critical facts required for correctness
- No important aspect of the question is left unaddressed
- The answer is not vague, partial, or ambiguous

Mark the answer as NOT sufficient if:
- Any key information is missing
- The answer is partial or incomplete
- The answer relies on assumptions or lacks clarity

### TASK
Given:
- The original question
- The current answer

Decide whether the answer is sufficient.

### INPUT
Original Question:
{query}

Current Answer:
{answer}

### OUTPUT FORMAT
Return ONLY a valid JSON object:
{{
  "is_sufficient": true or false,
  "reason": "brief explanation for the decision"
}}
""")

generate_final_answer_prompt = ChatPromptTemplate.from_template("""
You are an information retrieval assistant specialized in generating comprehensive answers.

Your goal is to provide a complete and accurate response to the original question based on the retrieved context.

### TASK
Given:
- The original question
- The retrieved context

You must:
1. Analyze the retrieved context thoroughly.
2. Synthesize the information to form a complete answer.
3. Ensure the answer directly addresses the original question.
4. Avoid including any information not supported by the retrieved context.
5. Be concise, clear and coherent in your response. 
6. Do NOT include any reasoning steps or explanations in the final answer.
7. The answer should be a direct response to the question, not a summary of the context.

### INPUT
Original Question:
{query}

Current Answer:
{answer}

Retrieved Context:
{context}

### OUTPUT FORMAT
Return ONLY a valid JSON object:
{{
  "answer": "your final answer here"
}}
""")