from langchain_core.tools import tool

VERBOSE_DESCRIPTION = (
    "Search the web for current information on any topic. This tool queries a search engine "
    "and returns relevant results including titles, URLs, and snippet summaries. Use this tool "
    "when you need to find up-to-date information that may not be in your training data, such as "
    "recent news, current market data, company information, technical documentation, or any "
    "factual query that benefits from real-time web data. The query parameter should be a clear, "
    "specific search string — avoid overly broad queries. Results are returned as a list of "
    "search result objects, each containing a title, URL, and text snippet. You can refine your "
    "search by making follow-up queries if the initial results aren't specific enough. "
    "Note: results may include sponsored content or outdated cached pages. "
    "Example: web_search(query='Acme Corp Q4 2025 earnings report')"
)

WEB_SEARCH_PRUNED = "Search the web for current information. Returns titles, URLs, and snippets."

WEB_SEARCH_ROUTING = "Web search for current information."


@tool(description=VERBOSE_DESCRIPTION)
def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        f"Search results for: '{query}'\n\n"
        f"1. [Market Analysis] Acme Corp Reports Strong Q4 Results\n"
        f"   URL: https://example.com/acme-q4\n"
        f"   Snippet: Acme Corp announced Q4 revenue of $2.3B, up 15% YoY. "
        f"Operating margin improved to 22% driven by cost optimization initiatives. "
        f"The company raised full-year guidance citing strong demand in enterprise segment.\n\n"
        f"2. [Industry Report] Enterprise Software Market Trends 2025-2026\n"
        f"   URL: https://example.com/enterprise-trends\n"
        f"   Snippet: The enterprise software market grew 12% in 2025, reaching $380B globally. "
        f"Key drivers include AI adoption, cloud migration, and cybersecurity spending. "
        f"Top performers include Acme Corp, TechFlow Inc, and DataBridge Systems.\n\n"
        f"3. [News] Acme Corp Expands APAC Operations\n"
        f"   URL: https://example.com/acme-apac\n"
        f"   Snippet: Acme Corp opened three new offices in Singapore, Tokyo, and Sydney "
        f"as part of its Asia-Pacific expansion strategy. The company plans to hire 500 "
        f"employees in the region by end of 2026.\n"
    )


web_search_tool = web_search
