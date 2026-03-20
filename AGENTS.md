# AGENTS.md

Instructions for AI coding agents (GitHub Copilot, Cursor, Claude Code) working on this project.

---

## Project Identity

This is a **multi-agent deep research system** built with LangGraph.
It is NOT a RAG pipeline. The core mechanic is: planner → parallel pro/con agents → judge.

Do not simplify this into a single-agent or single-retrieval system.

---

## Role and Boundaries

You are a backend Python agent working on an agentic AI system.

- Work exclusively inside `src/` and `tests/`
- Never modify `README.md` — that is maintained separately
- Never add new dependencies without updating `requirements.txt`
- Never hardcode API keys — always use `os.getenv()` with `.env`
- Never commit to `main` directly — all changes via feature branches

---

## Environment Setup (run before any task)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys manually
```

Required env vars:
```
DEEPSEEK_API_KEY      # from platform.deepseek.com
DEEPSEEK_BASE_URL     # https://api.deepseek.com
TAVILY_API_KEY        # from tavily.com
```

---

## Architecture Rules — Read Before Touching Any File

### State Management
- All shared data lives in `AgentState` (`src/graph/state.py`)
- Every agent node receives `AgentState` as input
- Every agent node must return a **dict** — not a modified state object
- Never mutate state in place — always return new values

```python
# CORRECT
def planner_node(state: AgentState) -> dict:
    return {"sub_questions": [...]}

# WRONG — never do this
def planner_node(state: AgentState) -> AgentState:
    state["sub_questions"] = [...]
    return state
```

### Parallel Execution
- Pro and Con agents run in parallel using LangGraph's `Send` API
- Do not change them to sequential — parallelism is intentional
- Both must write to separate state keys (`pro_evidence`, `con_evidence`)

```python
# This pattern must be preserved in workflow.py
from langgraph.types import Send

def route_to_researchers(state: AgentState):
    return [
        Send("pro_research", state),
        Send("con_research", state)
    ]
```

### LLM Calls
- Always use DeepSeek-V3 via OpenAI-compatible client:
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...]
)
```
- For JSON output from LLM, always add: `response_format={"type": "json_object"}`
- Always wrap LLM calls in try/except

### Tool Calls
- Web search is handled by Tavily in `src/tools/search.py`
- Use `TavilySearchResults(max_results=5)` — do not change max_results above 5
- Always decorate search functions with `@tool` from langchain

---

## File-by-File Responsibilities

| File | Purpose | Key constraint |
|---|---|---|
| `src/graph/state.py` | AgentState TypedDict | Never remove existing fields |
| `src/graph/workflow.py` | LangGraph graph definition | Preserve parallel execution |
| `src/agents/planner.py` | Query → sub_questions | Must return JSON array of strings |
| `src/agents/pro_agent.py` | Pro evidence search | Returns pro_evidence + pro_argument |
| `src/agents/con_agent.py` | Con evidence search | Returns con_evidence + con_argument |
| `src/agents/judge.py` | Verdict generation | Must return confidence_score as float 0-1 |
| `src/tools/search.py` | Tavily search tool | Keep @tool decorator |
| `src/api/main.py` | FastAPI endpoints | Keep /research and /health |
| `src/ui/app.py` | Streamlit UI | Keep 3-column layout |
| `run.py` | CLI entry point | Keep --query and --output args |

---

## Testing Commands

```bash
# Test individual agent
python -c "
from src.agents.planner import planner_node
result = planner_node({'query': 'Is coffee healthy?', 'messages': []})
print(result['sub_questions'])
"

# Test full pipeline
python run.py --query "Is remote work better for productivity?"

# Run test suite
pytest tests/ -v

# Run specific test
pytest tests/test_planner.py -v
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError: sub_questions` | Planner returning wrong key | Check return dict key matches exactly |
| `JSONDecodeError` from LLM | LLM not returning pure JSON | Add `response_format={"type": "json_object"}` |
| Parallel nodes not merging | Missing `Annotated[list, operator.add]` on list fields | Update AgentState to use operator.add for list fields |
| `TAVILY_API_KEY not set` | Missing env var | Check .env file exists and is loaded |
| `pro_evidence` is empty | Search returning no results | Try `search_depth="advanced"` in Tavily |

---

## Code Style

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all agent node functions
- PEP-8 formatting
- Max line length: 100 characters
- No print statements in production code — use `logging`

```python
import logging
logger = logging.getLogger(__name__)

def planner_node(state: AgentState) -> dict:
    """Break user query into 5-7 research sub-questions."""
    logger.info(f"Planning research for: {state['query']}")
    ...
```

---

## What NOT to Change

- Do not convert parallel agents to sequential
- Do not replace DeepSeek with GPT-4 (cost reasons)
- Do not change `AgentState` field names — downstream code depends on them
- Do not add streaming to the API without updating the Streamlit UI
- Do not add authentication to the API — this is a research demo

---

## Adding New Features

Before adding any new feature:
1. Check if it changes `AgentState` — if yes, update `state.py` first
2. Check if it changes the graph flow — if yes, update `workflow.py`
3. Add a corresponding test in `tests/`
4. Update `requirements.txt` if new package needed

---

*Last updated: March 2026*
*Project: Debate-Driven Research Agent*
*Author: Nikhil Kumar, IIIT Lucknow*
