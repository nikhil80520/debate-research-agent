import json
import logging
import os
import re

from dotenv import load_dotenv

from src.graph.state import AgentState

load_dotenv()

logger = logging.getLogger(__name__)


def judge_node(state: AgentState) -> dict:
    """Compare both sides and deliver an impartial verdict with confidence."""
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
                        "Return ONLY raw JSON, no markdown, no backticks: "
                        "{\"verdict\": \"...\", \"confidence_score\": 0.75, "
                        "\"stronger_side\": \"pro\", "
                        "\"key_uncertainties\": [\"...\"]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {state['query']}\n\n"
                        f"PRO:\n{state['pro_argument']}\n\n"
                        f"CON:\n{state['con_argument']}"
                    ),
                },
            ],
        )

        text = (response.choices[0].message.content or "").strip()
        parsed: dict = {}
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = {}

        verdict = str(parsed.get("verdict", "Unable to generate")).strip()
        confidence_raw = parsed.get("confidence_score", 0.0)

        try:
            confidence_score = float(confidence_raw)
        except (TypeError, ValueError):
            confidence_score = 0.0

        confidence_score = max(0.0, min(1.0, confidence_score))

        final_report = (
            f"## Query\n{state['query']}\n\n"
            f"## Pro Summary\n{state['pro_argument']}\n\n"
            f"## Con Summary\n{state['con_argument']}\n\n"
            f"## Verdict\n{verdict}\n\n"
            f"## Confidence\n{confidence_score:.2f}"
        )

        return {
            "verdict": verdict,
            "confidence_score": confidence_score,
            "final_report": final_report,
        }
    except Exception as exc:
        logger.error("Judge failed: %s", exc)
        return {
            "verdict": "Unable to generate",
            "confidence_score": 0.0,
            "final_report": "",
        }
