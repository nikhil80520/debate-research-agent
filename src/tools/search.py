import logging
import os

from langchain_tavily import TavilySearch
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY not set")
            return ""

        search_tool = TavilySearch(max_results=5)
        raw_results = search_tool.invoke(query)
        results: list = []
        if isinstance(raw_results, dict):
            candidate = raw_results.get("results", [])
            if isinstance(candidate, list):
                results = candidate
        elif isinstance(raw_results, list):
            results = raw_results
        elif isinstance(raw_results, str):
            return raw_results

        formatted_results: list[str] = []
        for result in results:
            if isinstance(result, dict):
                url = str(result.get("url", ""))
                content = str(result.get("content", ""))
                formatted_results.append(f"Source: {url}\n{content}")
            else:
                formatted_results.append(str(result))

        return "\n\n".join(formatted_results)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return ""
