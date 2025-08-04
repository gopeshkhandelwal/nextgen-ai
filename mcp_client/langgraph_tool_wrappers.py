# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

from langchain_core.tools import tool
from mcp import ClientSession

client_session: ClientSession = None  # Will be set by main client

def build_tool_wrappers():
    '''Builds tool wrappers for the MCP client session.'''
    tools = []

    @tool
    async def city_weather(city: str) -> str:
        """Fetch current weather for a given city using the OpenWeather API."""
        print(f"mcp_client: Fetching weather for city: {city}")
        result = await client_session.call_tool("city_weather", {"city": city})
        return result.content[0].text
   
    @tool
    async def list_itac_products() -> str:
        """
        Fetch and return a list of available ITAC products.

        This tool queries the ITAC API to retrieve product information and extracts their
        names for listing.

        Requirements:
        - The environment variable ITAC_API_TOKEN must be set.
        - The endpoint URL must be provided via ITAC_PRODUCTS.

        Returns:
        - A newline-separated string of ITAC product names, or an error message if none are found.
        """
        result = await client_session.call_tool("list_itac_products")
        return result.content[0].text

    @tool
    async def document_qa(question: str, search_method: str = "hybrid", use_reranking: bool = False) -> str:
        """
        Answer questions SPECIFICALLY about ITAC Compute gRPC APIs, documentation, endpoints, authentication methods, and technical details.

        This tool uses advanced hybrid retrieval combining BM25 keyword search and semantic vector search
        to find relevant documents from ITAC gRPC API documentation, with optional reranking for enhanced accuracy.

        ONLY use this tool for questions about:
        - ITAC Compute gRPC APIs and their specifications
        - API endpoints, authentication, and integration
        - gRPC service operations and protobuf definitions
        - Vault integration and mTLS authentication
        - Technical documentation and API usage
        
        DO NOT use this tool for:
        - Weather information (use city_weather instead)
        - General questions not related to ITAC APIs
        - ITAC products (use list_itac_products instead)

        Args:
            question: The question about ITAC APIs to answer
            search_method: Search method - "hybrid" (default), "semantic", or "keyword"  
            use_reranking: Whether to use reranking for better relevance (slower but more accurate)
        """
        # More sophisticated trigger detection
        reranking_indicators = [
            "detailed", "comprehensive", "thorough", "precise", "exactly", "best",
            "explain in detail", "step-by-step", "how exactly", "elaborate",
            "complete guide", "full explanation", "in-depth"
        ]
        
        use_reranking = any(indicator in question.lower() for indicator in reranking_indicators)
        print(f"mcp_client: Using reranking: {use_reranking} for question: {question}")
        result = await client_session.call_tool("document_qa", {
            "question": question,
            "search_method": search_method,
            "use_reranking": use_reranking
        })
        return result.content[0].text
   

    tools.extend([city_weather, list_itac_products, document_qa])
    return tools
