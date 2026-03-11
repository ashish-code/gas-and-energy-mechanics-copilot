"""
Module: tools/web_search_tool.py

PURPOSE:
    A CrewAI tool that searches the web for real-time regulatory information
    when the FAISS index may be out of date. The FAISS index is a point-in-time
    snapshot of 49 CFR regulations and Wikipedia articles built at index time.
    PHMSA continuously issues:
      - Advisory bulletins (ADB) with urgent guidance
      - Final rules amending 49 CFR text
      - Enforcement actions against specific operators
      - Integrity management directives

    None of these post-index documents are in FAISS. This tool fills that gap
    by searching trusted regulatory websites in real time via Tavily API.

ARCHITECTURE POSITION:
    This tool is given to retrieval_specialist in FullPipelineCrew.
    It is only invoked when the router_agent sets requires_web_search=True,
    which typically happens for queries about:
      - Recent enforcement actions ("Was Operator X fined in 2024?")
      - New final rules ("Does the 2023 gas pipeline safety rule apply to me?")
      - PHMSA advisory bulletins ("What does ADB-2024-01 say?")

    Flow:
        router_agent (requires_web_search=True)
             ↓
        retrieval_specialist (reads router output from context)
             ↓ calls RegulatoryWebSearchTool
        Tavily API → filtered to regulatory domains only
             ↓
        Returns structured results for synthesis_task

WHY TAVILY over alternatives:
    - vs SerpAPI: SerpAPI returns raw Google search results (URLs, snippets,
      page titles). The agent must then fetch and parse each page. Tavily
      returns pre-extracted, cleaned text optimised for LLM RAG consumption —
      no additional fetching required.
    - vs Bing Search API: Bing returns the same raw results as SerpAPI. Tavily's
      "answer mode" synthesises a single answer from top results, and its
      "search mode" returns cleaned full-text passages with relevance scores.
    - vs DuckDuckGo (ddg-search, free): No API key, but severely rate-limited,
      no structured output, and no domain filtering. Unreliable for production.
    - vs Exa (formerly Metaphor): Exa specialises in technical/academic content.
      Tavily covers regulatory and government sites more comprehensively.
    - vs direct URL fetch: PHMSA pages are JavaScript-rendered, making direct
      requests with urllib unreliable. Tavily handles JS-rendered pages.

DOMAIN FILTERING:
    Restricts results to authoritative regulatory sources to prevent the agent
    from citing low-quality or misleading secondary sources:
      - phmsa.dot.gov    : Primary PHMSA source (bulletins, final rules, enforcement)
      - ecfr.gov         : Current Code of Federal Regulations text
      - ferc.gov         : Federal Energy Regulatory Commission orders
      - gpo.gov          : Government Publishing Office (Federal Register)
      - congress.gov     : Pipeline Safety Improvement Act legislative text

API KEY SETUP:
    Set the TAVILY_API_KEY environment variable. Get a free API key at:
    https://app.tavily.com (includes 1,000 free searches/month)
    Example: export TAVILY_API_KEY=tvly-...

GRACEFUL DEGRADATION:
    If TAVILY_API_KEY is not set, the tool returns an informative message
    instead of raising an exception. This allows the crew to continue with
    FAISS-only retrieval without crashing.
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:  # pragma: no cover
    from crewai_tools import BaseTool  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# Authoritative regulatory domains — searches are restricted to these sites.
# This prevents the agent from citing blog posts, industry association summaries,
# or other secondary sources that may paraphrase regulations inaccurately.
REGULATORY_DOMAINS = [
    "phmsa.dot.gov",
    "ecfr.gov",
    "ferc.gov",
    "gpo.gov",
    "congress.gov",
    "transportation.gov",
]

# Maximum number of result documents to return per search call.
# Higher = more context but more tokens consumed. 5 is a good default.
MAX_RESULTS = 5

# Maximum character length for each result's content snippet.
# Tavily can return very long full-text extracts; we cap to avoid token explosion.
MAX_CONTENT_CHARS = 2000


class WebSearchInput(BaseModel):
    """Input schema for RegulatoryWebSearchTool."""

    query: str = Field(
        description=(
            "The search query to issue. Be specific — include CFR section numbers, "
            "bulletin IDs, or operator names for best results. "
            "Example: 'PHMSA ADB-2024-01 advisory bulletin natural gas' or "
            "'49 CFR 192 final rule 2023 gas gathering pipelines'"
        )
    )
    include_domains: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of domains to include (overrides the default regulatory domain filter). "
            "Leave empty to use the default regulatory domain filter."
        ),
    )


class RegulatoryWebSearchTool(BaseTool):
    """
    Searches authoritative regulatory websites for real-time pipeline safety information.

    This tool is the complement to RAGSearchTool: while RAGSearchTool searches the
    pre-built FAISS index (point-in-time snapshot), this tool searches the live web
    restricted to trusted government and regulatory sites.

    Use this tool when:
      - The router_agent has set requires_web_search=True.
      - The question explicitly mentions recent events, bulletins, or final rules.
      - The FAISS search returned no relevant results for a known regulatory topic.

    Do NOT use this tool for:
      - Basic regulatory lookups (§192.505, §195.410) → use RAGSearchTool instead.
      - General engineering background → the FAISS index has Wikipedia articles.

    Returns:
        A formatted string with search results, each including:
          - Source URL and title
          - Relevance score [0.0, 1.0]
          - Extracted text content (up to MAX_CONTENT_CHARS characters)
    """

    name: str = "search_regulatory_web"
    description: str = (
        "Search authoritative regulatory websites (phmsa.dot.gov, ecfr.gov, ferc.gov, gpo.gov) "
        "for real-time pipeline safety information. Use this when the FAISS index may be "
        "out of date — for recent enforcement actions, advisory bulletins, or new final rules. "
        "Returns cleaned text passages from official government sources."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str, include_domains: list[str] | None = None) -> str:
        """
        Execute a Tavily web search restricted to regulatory domains.

        The search uses Tavily's "search" mode which returns raw passages from
        pages rather than a synthesised answer. This gives the retrieval_specialist
        access to the actual regulatory text rather than an LLM's interpretation of it.

        Args:
            query:          Search query string.
            include_domains: Optional domain whitelist (overrides REGULATORY_DOMAINS).

        Returns:
            Formatted string with numbered search results, or an error message
            if the API key is missing or the search fails.
        """
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            log.warning("TAVILY_API_KEY not set — web search unavailable")
            return (
                "Web search is not available (TAVILY_API_KEY environment variable not set). "
                "Falling back to FAISS index only. For real-time regulatory information, "
                "set TAVILY_API_KEY=tvly-... (free tier: 1,000 searches/month at app.tavily.com)."
            )

        # Lazy import: tavily-python is an optional dependency
        try:
            from tavily import TavilyClient
        except ImportError:
            log.warning("tavily-python package not installed — web search unavailable")
            return (
                "Web search unavailable: tavily-python package not installed. "
                "Run: uv add tavily-python"
            )

        domains = include_domains or REGULATORY_DOMAINS
        log.info("Executing regulatory web search", extra={"query": query, "domains": domains})

        try:
            client = TavilyClient(api_key=api_key)

            # `search` mode: returns raw passages from multiple pages.
            # `include_raw_content=False`: we use content (cleaned text), not raw HTML.
            # `max_results`: number of distinct pages/documents to return.
            response = client.search(
                query=query,
                search_depth="advanced",   # "basic" = faster; "advanced" = more thorough
                include_domains=domains,
                max_results=MAX_RESULTS,
                include_answer=False,      # we want raw passages, not Tavily's synthesised answer
                include_raw_content=False,
            )

            results = response.get("results", [])
            if not results:
                log.info("Web search returned no results", extra={"query": query})
                return (
                    f"No results found on regulatory domains for query: '{query}'. "
                    "The information may not be available in recent publications or "
                    "may be specific to the FAISS-indexed regulations."
                )

            # Format results for the agent to read
            formatted_parts = [
                f"Web Search Results for: '{query}'\n"
                f"Sources restricted to: {', '.join(domains)}\n"
                f"{'─' * 60}"
            ]

            for i, result in enumerate(results, start=1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                score = result.get("score", 0.0)
                content = result.get("content", "")

                # Truncate long content to avoid token explosion in the agent context
                if len(content) > MAX_CONTENT_CHARS:
                    content = content[:MAX_CONTENT_CHARS] + " [... content truncated ...]"

                formatted_parts.append(
                    f"\n[{i}] {title}\n"
                    f"    URL:   {url}\n"
                    f"    Score: {score:.3f}\n"
                    f"    Content:\n{content}\n"
                )

            formatted_parts.append(
                f"\n{'─' * 60}\n"
                f"Note: These results are from live web search and may supersede "
                f"the indexed 49 CFR text in the FAISS database."
            )

            result_text = "\n".join(formatted_parts)
            log.info(
                "Web search completed",
                extra={"query": query, "result_count": len(results)},
            )
            return result_text

        except Exception as e:
            log.warning("Tavily web search failed", extra={"query": query, "error": str(e)})
            return (
                f"Web search failed for query '{query}': {e}. "
                "Falling back to FAISS index results only."
            )
