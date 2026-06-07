from langchain_community.tools.tavily_search import TavilySearchResults

def get_search_tool():
    """
    Configures and returns the Tavily Search tool.
    """
    # Initialize tool returning up to 5 results with the new simplified class
    tool = TavilySearchResults(
        max_results=5,
        description="Search the web for live, up-to-date information, news, and current events."
    )
    return tool