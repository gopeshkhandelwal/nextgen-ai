import os
import logging
import httpx

logger = logging.getLogger("idc_tools")
IDC_API_TOKEN = os.getenv("IDC_API_TOKEN")

def register_tools(mcp):
    @mcp.tool()
    async def list_idc_pools() -> str:
        logger.info("Tool called: list_idc_pools")
        if not IDC_API_TOKEN:
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."
        url = "https://compute-us-region-5-api.cloud.intel.com/v1/fleetadmin/computenodepools"
        logger.info(f"URL: {url}")
        headers = {"Authorization": f"Bearer {IDC_API_TOKEN}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                pool_names = [item["poolName"] for item in data.get("computeNodePools", [])]
                response = "\n".join(pool_names or ["No pool found."])
                logger.info(f"list_idc_pools response: {response}")
                return response
        except Exception as e:
            logger.exception(f"Error fetching IDC pools: {e}")
            return "Failed to retrieve IDC pools."
