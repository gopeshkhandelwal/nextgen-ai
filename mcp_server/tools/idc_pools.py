import os
import logging
import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

IDC_API_TOKEN = os.getenv("IDC_API_TOKEN")
IDC_API_POOLS = os.getenv("IDC_API_POOLS")

def register_tools(mcp):
    @mcp.tool()
    async def list_idc_pools() -> str:
        logger.info("Tool called: list_idc_pools")
        if not IDC_API_TOKEN:
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."
        
        logger.info(f"IDC_API_POOLS: {IDC_API_POOLS}")
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
