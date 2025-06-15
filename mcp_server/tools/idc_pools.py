import os
import logging
import httpx
from dotenv import load_dotenv

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read IDC API configuration from environment
IDC_API_TOKEN = os.getenv("IDC_API_TOKEN")
IDC_API_POOLS = os.getenv("IDC_API_POOLS")

def register_tools(mcp):
    """
    Register IDC-related tools with the MCP server.
    """

    @mcp.tool()
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
        logger.info("Tool called: list_idc_pools")

        if not IDC_API_TOKEN:
            logger.error("IDC API token not configured. Please set IDC_API_TOKEN in your environment.")
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."
        
        if not IDC_API_POOLS:
            logger.error("IDC API pools endpoint not configured. Please set IDC_API_POOLS in your environment.")
            return "IDC API pools endpoint not configured. Please set IDC_API_POOLS in your environment."

        headers = {"Authorization": f"Bearer {IDC_API_TOKEN}"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(IDC_API_POOLS, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                pools = data.get("computeNodePools")

                if not pools:
                    logger.warning("No computeNodePools found in response")
                    return "No compute pools returned by the IDC API."

                pool_names = [item["poolName"] for item in pools]
                output = "\n".join(pool_names)
                logger.info(f"list_idc_pools response:\n{output}")
                return output
        except Exception as e:
            logger.exception(f"Error fetching IDC pools: {e}")
            return "Failed to retrieve IDC pools."
