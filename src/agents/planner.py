import json
import logging
import os
import re

from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def planner_node(state: AgentState) -> dict:
    """Break user query into 5 balanced pro/con research sub-questions."""
    logger.info("Planning sub-questions for query: %s", state["query"])

    try:
        from cerebras.cloud.sdk import Cerebras

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not set")

        client = Cerebras(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research planner. Break the user query into exactly "
                        "5 focused sub-questions that collectively cover both PRO and "
                        "CON perspectives. Return ONLY a valid JSON array of 5 strings."
                    ),
                },
                {"role": "user", "content": state["query"]},
            ],
        )

        text = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            sub_questions = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            sub_questions = [state["query"]]

        if len(sub_questions) != 5:
            raise ValueError("Planner response must contain exactly 5 strings")

        return {"sub_questions": sub_questions}
    except Exception as exc:
        logger.error("Planner failed: %s", exc)
        return {"sub_questions": [state["query"]]}
