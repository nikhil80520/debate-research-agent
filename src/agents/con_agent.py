import json
import logging
import os
import re

from dotenv import load_dotenv

from src.graph.state import AgentState
from src.tools.search import web_search

load_dotenv()

logger = logging.getLogger(__name__)
MODEL_NAME = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
FALLBACK_MODEL_NAME = os.getenv("CEREBRAS_FALLBACK_MODEL", "llama3.1-8b")


def con_research_node(state: AgentState) -> dict:
    """Research and synthesize evidence against the main claim."""
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

        response = None
        models_to_try = [MODEL_NAME]
        if FALLBACK_MODEL_NAME != MODEL_NAME:
            models_to_try.append(FALLBACK_MODEL_NAME)

        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
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
                logger.info("Con agent used model: %s", model_name)
                break
            except Exception as model_exc:
                error_text = str(model_exc).lower()
                if (
                    "model_not_found" in error_text
                    or "does not exist" in error_text
                    or "do not have access" in error_text
                ) and model_name != models_to_try[-1]:
                    logger.warning("Model unavailable for con agent: %s", model_name)
                    continue
                raise

        if response is None:
            raise RuntimeError("Failed to get response from any configured model")

        text = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            argument = data.get("argument", text)
        else:
            argument = text

        con_argument = str(argument).strip()
        con_evidence = evidence_items

        return {"con_evidence": con_evidence, "con_argument": con_argument}
    except Exception as exc:
        logger.error("Con research failed: %s", exc)
        return {"con_evidence": [], "con_argument": ""}
