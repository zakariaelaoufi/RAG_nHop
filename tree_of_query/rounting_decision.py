from dataclasses import dataclass


@dataclass
class RoutingDecision:
    route: str  # "SINGLE_HOP" or "MULTI_HOP"
    confidence: float  # Confidence score between 0 and 1
    reason: str  # Short justification for the decision
