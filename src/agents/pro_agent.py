import json
import logging
import os
import re

from dotenv import load_dotenv

from src.graph.state import AgentState
from src.tools.search import web_search

load_dotenv()

logger = logging.getLogger(__name__)


def pro_research_node(state: AgentState) -> dict:
    """Research and synthesize evidence supporting the main claim."""
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
                    "You are a research agent finding evidence SUPPORTING the claim. "
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
                data: dict = {}
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        data = {}

                tool_name = str(data.get("name", "")).strip()
                tool_args = data.get("arguments", {})
                if tool_name == "web_search" and isinstance(tool_args, dict):
                    query = str(tool_args.get("query", "")).strip()
                    if query:
                        result = web_search.invoke(query)
                        result_truncated = str(result)[:500]
                        evidence_items.append(
                            {
                                "source": query,
                                "content": result_truncated,
                                "url": "",
                            }
                        )
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": result_truncated})
                        continue

                argument = data.get("argument", text) if data else text
                return {
                    "pro_evidence": evidence_items,
                    "pro_argument": str(argument),
                }

        final_response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Based on all the evidence gathered, provide your final "
                        "synthesis as JSON: {\"argument\": \"your complete "
                        "synthesis here\"}"
                    ),
                }
            ],
        )
        text = (final_response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                argument = data.get("argument", text)
            except json.JSONDecodeError:
                argument = text
        else:
            argument = text
        return {"pro_evidence": evidence_items, "pro_argument": str(argument)}
    except Exception as exc:
        logger.error("Pro research failed: %s", exc)
        return {"pro_evidence": [], "pro_argument": ""}
