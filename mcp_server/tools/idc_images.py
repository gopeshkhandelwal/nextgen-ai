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
IDC_API_IMAGES = os.getenv("IDC_API_IMAGES")

def register_tools(mcp):
    """
    Register IDC-related tools with the MCP server.
    """

    @mcp.tool()
    async def list_idc_images() -> str:
        """
        Fetch and return a list of available IDC machine image names.

        This tool queries the IDC Compute API using an authenticated request and returns the
        metadata.name of each image found in the response.

        Requirements:
        - The environment variable IDC_API_TOKEN must be set.
        - The endpoint URL must be provided via IDC_API_IMAGES.

        Returns:
        - A newline-separated string of image names, or an error message if the request fails.
        """
        logger.info("Tool called: list_idc_images")

        if not IDC_API_TOKEN:
            logger.error("IDC API token not configured. Please set IDC_API_TOKEN in your environment.")
            return "IDC API token not configured. Please set IDC_API_TOKEN in your environment."

        if not IDC_API_IMAGES:
            logger.error("IDC API images endpoint not configured. Please set IDC_API_IMAGES in your environment.")
            return "IDC API images endpoint not configured. Please set IDC_API_IMAGES in your environment."

        headers = {"Authorization": f"Bearer {IDC_API_TOKEN}"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(IDC_API_IMAGES, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                image_names = [item["metadata"]["name"] for item in data.get("items", [])]
                if image_names:
                    logger.info("Fetched %d IDC images.", len(image_names))
                    return "\n".join(image_names)
                else:
                    logger.info("No images found in IDC API response.")
                    return "No images found."
        except Exception as e:
            logger.exception("Error fetching IDC images")
            return "Failed to retrieve IDC image list."
