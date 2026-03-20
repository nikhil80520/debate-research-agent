import json
import logging
import os
import re

from dotenv import load_dotenv

from src.graph.state import AgentState

load_dotenv()

logger = logging.getLogger(__name__)
MODEL_NAME = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
FALLBACK_MODEL_NAME = os.getenv("CEREBRAS_FALLBACK_MODEL", "llama3.1-8b")


def judge_node(state: AgentState) -> dict:
    """Compare both sides and deliver an impartial verdict with confidence."""
    try:
        from cerebras.cloud.sdk import Cerebras

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not set")

        client = Cerebras(api_key=api_key)
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
                logger.info("Judge used model: %s", model_name)
                break
            except Exception as model_exc:
                error_text = str(model_exc).lower()
                if (
                    "model_not_found" in error_text
                    or "does not exist" in error_text
                    or "do not have access" in error_text
                ) and model_name != models_to_try[-1]:
                    logger.warning("Model unavailable for judge: %s", model_name)
                    continue
                raise

        if response is None:
            raise RuntimeError("Failed to get response from any configured model")

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
