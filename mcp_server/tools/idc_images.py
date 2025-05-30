import os
import logging
import httpx

logger = logging.getLogger("idc_tools")
IDC_API_TOKEN = os.getenv("IDC_API_TOKEN")

def register_tools(mcp):
    @mcp.tool()
    async def list_idc_images() -> str:
        logger.info("Tool called: list_idc_images")
        if not IDC_API_TOKEN:
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."
        url = "https://compute-us-region-1-api.cloud.intel.com/v1/machineimages"
        headers = {"Authorization": f"Bearer {IDC_API_TOKEN}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                image_names = [item["metadata"]["name"] for item in data.get("items", [])]
                return "\n".join(image_names or ["No images found."])
        except Exception:
            logger.exception("Error fetching IDC images")
            return "Failed to retrieve IDC image list."
