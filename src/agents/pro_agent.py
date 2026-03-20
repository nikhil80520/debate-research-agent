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

        evidence_items: list[dict] = []
        compiled_search_notes: list[str] = []

        for question in state["sub_questions"]:
            search_result = web_search.invoke(question)
            search_result = search_result[:300]
            if not search_result:
                continue
            compiled_search_notes.append(f"Question: {question}\n{search_result}")
            evidence_items.append(
                {
                    "source": question,
                    "content": search_result,
                    "url": "",
                }
            )

        all_results = "\n\n".join(compiled_search_notes)
        findings_text = all_results[:1500]

        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research agent. Respond with ONLY a single valid JSON "
                        "object. No markdown, no code blocks, no explanation. Just raw "
                        "JSON like this: {\"argument\": \"your analysis here\", "
                        "\"key_points\": [\"point1\", \"point2\"]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Main query: {state['query']}\n\n"
                        f"Search findings:\n\n{findings_text}"
                    ),
                },
            ],
        )

        text = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            argument = data.get("argument", text)
        else:
            argument = text

        pro_argument = str(argument).strip()
        pro_evidence = evidence_items

        return {"pro_evidence": pro_evidence, "pro_argument": pro_argument}
    except Exception as exc:
        logger.error("Pro research failed: %s", exc)
        return {"pro_evidence": [], "pro_argument": ""}
