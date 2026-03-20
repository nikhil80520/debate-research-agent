# Copilot Instructions — Debate-Driven Research Agent

## Project overview

This is a multi-agent deep research system built with LangGraph.
A Planner agent breaks a query into sub-questions.
Pro and Con agents run in parallel searching evidence for both sides.
A Judge agent synthesizes both and delivers a verdict with confidence score.

## Tech stack

- Python 3.11+
- LangGraph 0.2+ for agent orchestration
- DeepSeek-V3 via OpenAI-compatible API as the LLM
- Tavily API for web search
- FastAPI for the REST API
- Streamlit for the demo UI

## Hard rules — never break these

- Never hardcode API keys — always use `os.getenv()`
- Never mutate AgentState in place — always return a new dict
- Never convert parallel Pro/Con agents to sequential
- Never use GPT-4 or Claude — use DeepSeek-V3 (`model="deepseek-chat"`)
- All LLM calls that need JSON output must include `response_format={"type": "json_object"}`
- Always wrap LLM and API calls in try/except

## Code style

- Type hints on every function signature
- Docstrings on every agent node function
- Use `logging` not `print` in production code
- Max line length: 100 characters
- PEP-8 formatting

## File boundaries

- Agent logic lives only in `src/agents/`
- Tool definitions live only in `src/tools/`
- Graph and state live only in `src/graph/`
- Never mix agent logic into the API or UI files