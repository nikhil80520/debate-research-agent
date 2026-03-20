import json
import logging
import os
import re

from dotenv import load_dotenv

from src.graph.state import AgentState
from src.tools.search import web_search

load_dotenv()

logger = logging.getLogger(__name__)


def con_research_node(state: AgentState) -> dict:
    """Research and synthesize evidence against the main claim."""
    try:
        from cerebras.cloud.sdk import Cerebras

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not set")

        client = Cerebras(api_key=api_key)

        max_iterations = 3
        evidence_items: list[dict] = []
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a research agent finding evidence AGAINST the claim. "
                    "You have access to web_search tool. In each step: decide if you need "
                    "more information, call web_search if yes, or return final JSON if you "
                    "have enough evidence. When done, return ONLY raw JSON: "
                    "{\"argument\": \"your synthesis\", \"sufficient\": true}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Main query: {state['query']}\n\n"
                    "Starting research directions:\n"
                    f"{json.dumps(state['sub_questions'])}"
                ),
            },
        ]

        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model="llama3.1-8b",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search web for information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                },
                                "required": ["query"],
                            },
                        },
                    }
                ],
                tool_choice="auto",
                messages=messages,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                assistant_tool_calls: list[dict] = []
                for tool_call in msg.tool_calls:
                    assistant_tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": assistant_tool_calls,
                    }
                )

                for tool_call in msg.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                        query = str(args.get("query", "")).strip()
                    except json.JSONDecodeError:
                        query = ""

                    if not query:
                        result_truncated = ""
                    else:
                        result = web_search.invoke(query)
                        result_truncated = str(result)[:500]
                        evidence_items.append(
                            {
                                "source": query,
                                "content": result_truncated,
                                "url": "",
                            }
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_truncated,
                        }
                    )
            else:
                text = (msg.content or "").strip()
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    argument = data.get("argument", text)
                else:
                    argument = text
                return {
                    "con_evidence": evidence_items,
                    "con_argument": str(argument),
                }

        return {
            "con_evidence": evidence_items,
            "con_argument": "Research complete based on gathered evidence",
        }
    except Exception as exc:
        logger.error("Con research failed: %s", exc)
        return {"con_evidence": [], "con_argument": ""}
