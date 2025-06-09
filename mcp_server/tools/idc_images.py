import os
import logging
import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

IDC_API_TOKEN = os.getenv("IDC_API_TOKEN")
IDC_API_IMAGES = os.getenv("IDC_API_IMAGES")

def register_tools(mcp):
    @mcp.tool()
    async def list_idc_images() -> str:
        logger.info("Tool called: list_idc_images")
        if not IDC_API_TOKEN:
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."

        logger.info(f"IDC_API_IMAGES: {IDC_API_IMAGES}")
        headers = {"Authorization": f"Bearer {IDC_API_TOKEN}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(IDC_API_IMAGES, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                image_names = [item["metadata"]["name"] for item in data.get("items", [])]
                return "\n".join(image_names or ["No images found."])
        except Exception as e:
            logger.exception("Error fetching IDC images", e)
            return "Failed to retrieve IDC image list."
