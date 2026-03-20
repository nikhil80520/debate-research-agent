from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph.workflow import research_graph

load_dotenv()

app = FastAPI(title="Debate-Driven Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/research")
def research(payload: ResearchRequest) -> dict:
    """Run the research graph and return verdict, confidence, and report."""
    result = research_graph.invoke(
        {
            "query": payload.query,
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
    )

    return {
        "verdict": result.get("verdict", ""),
        "confidence_score": float(result.get("confidence_score", 0.0)),
        "final_report": result.get("final_report", ""),
    }
