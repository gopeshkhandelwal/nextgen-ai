from langchain_core.tools import tool
from mcp import ClientSession

client_session: ClientSession = None  # Will be set by main client

def build_tool_wrappers():
    '''Builds tool wrappers for the MCP client session.
    Returns a list of tool functions that can be used in the LangGraph agent.
    '''
    tools = []

    @tool
    async def city_weather(city: str) -> str:
        """Fetch current weather for a given city using the OpenWeather API."""
        print(f"mcp_client: Fetching weather for city: {city}")
        result = await client_session.call_tool("city_weather", {"city": city})
        return result.content[0].text

    @tool
    async def list_idc_pools() -> str:
        """
        Fetch and return a list of available IDC compute node pool names.

        This tool queries the IDC Compute API to retrieve computeNodePools and extracts their
        poolName field for listing.

        Requirements:
        - The environment variable IDC_API_TOKEN must be set.
        - The endpoint URL must be provided via IDC_API_POOLS.

        Returns:
        - A newline-separated string of compute pool names, or an error message if none are found.
        """
        result = await client_session.call_tool("list_idc_pools")
        return result.content[0].text

    @tool
    async def idc_grpc_api(question: str) -> str:
        """
        Answer questions about IDC Compute gRPC APIs, endpoints, authentication methods, Vault integration,

        Returns a synthesized response using an LLM and the retrieved IDC gRPC API documentation.

        Supported topics include:
        - Public/private IDC gRPC APIs and their Swagger or protobuf definitions
        - Authentication using mTLS and Vault annotations
        - Using grpcurl for testing or exploration
        - Service operations like InstanceService, VNetService, MachineImageService, etc.
        """
        result = await client_session.call_tool("document_qa", {"question": question})
        return result.content[0].text

    tools.extend([city_weather, list_idc_pools, idc_grpc_api])
    return tools
