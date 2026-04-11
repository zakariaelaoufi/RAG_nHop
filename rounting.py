from prompt import rag_routing_prompt
from rounting_decision import RoutingDecision
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse
import json
import os

load_dotenv()


def determine_routing(query: str) -> RoutingDecision:
    prompt = rag_routing_prompt.format_messages(query=query)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Parse the JSON response
    try:
        decision_data = json.loads(response_text)
        return RoutingDecision(
            route=decision_data["route"],
            confidence=decision_data["confidence"],
            reason=decision_data["reason"]
        )
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid response format: {response_text}") from e
    

def main():
    parser = argparse.ArgumentParser(description="Determine routing for a given query")
    parser.add_argument("query", help="The query to route")
    args = parser.parse_args()

    decision = determine_routing(args.query)
    print(f"Routing Decision: {decision.route}, Confidence: {decision.confidence}, Reason: {decision.reason}")


if __name__ == "__main__":
    main()