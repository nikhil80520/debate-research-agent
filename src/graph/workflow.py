from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.con_agent import con_research_node
from src.agents.judge import judge_node
from src.agents.planner import planner_node
from src.agents.pro_agent import pro_research_node
from src.graph.state import AgentState


def route_to_researchers(state: AgentState) -> list[Send]:
    """Route planner output to pro and con researchers in parallel."""
    return [
        Send("pro_research", state),
        Send("con_research", state),
    ]


graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("pro_research", pro_research_node)
graph.add_node("con_research", con_research_node)
graph.add_node("judge", judge_node)

graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", route_to_researchers)
graph.add_edge("pro_research", "judge")
graph.add_edge("con_research", "judge")
graph.add_edge("judge", END)

research_graph = graph.compile()
