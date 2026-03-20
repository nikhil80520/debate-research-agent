import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.graph.workflow import research_graph

load_dotenv()


def build_base_state(query: str) -> dict:
    """Build the initial state payload for graph invocation."""
    return {
        "query": query,
        "messages": [],
        "sub_questions": [],
        "pro_evidence": [],
        "con_evidence": [],
        "pro_argument": "",
        "con_argument": "",
        "verdict": "",
        "confidence_score": 0.0,
        "final_report": "",
    }


def main() -> None:
    """Run the research workflow from the command line."""
    parser = argparse.ArgumentParser(description="Debate-Driven Research CLI")
    parser.add_argument("--query", required=True, help="Research query to analyze")
    parser.add_argument("--output", default="report.md", help="Output markdown file path")
    args = parser.parse_args()

    result = research_graph.invoke(build_base_state(args.query))

    output_path = Path(args.output)
    output_path.write_text(result.get("final_report", ""), encoding="utf-8")

    print(f"Verdict: {result.get('verdict', '')}")
    print(f"Confidence: {result.get('confidence_score', 0.0)}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
