import logging
import os

from langchain_community.tools.tavily_search import TavilySearchResults
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

        search_tool = TavilySearchResults(max_results=5, tavily_api_key=api_key)
        results = search_tool.invoke(query)

        formatted_results: list[str] = []
        for result in results:
            url = result.get("url", "")
            content = result.get("content", "")
            formatted_results.append(f"Source: {url}\n{content}")

        return "\n\n".join(formatted_results)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return ""
