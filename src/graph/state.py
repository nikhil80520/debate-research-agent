from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    sub_questions: list[str]
    pro_evidence: list[dict]  # [{source, content, url}]
    con_evidence: list[dict]
    pro_argument: str
    con_argument: str
    verdict: str
    confidence_score: float  # 0.0 to 1.0
    final_report: str
    messages: Annotated[list, add_messages]
