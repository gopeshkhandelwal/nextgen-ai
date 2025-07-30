import os
import logging
import httpx
from dotenv import load_dotenv

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read ITAC API configuration from environment
ITAC_API_TOKEN = os.getenv("ITAC_API_TOKEN")
ITAC_PRODUCTS = os.getenv("ITAC_PRODUCTS")

def register_tools(mcp):
    """
    Register ITAC-related tools with the MCP server.
    """

    @mcp.tool(name="list_itac_products")
    async def list_itac_products() -> str:
        """
        Fetch and return a list of available ITAC products.

        This tool queries the ITAC API to retrieve products and extracts their
        names for listing.
        productName field for listing.

        Requirements:
        - The environment variable ITAC_API_TOKEN must be set.
        - The endpoint URL must be provided via ITAC_PRODUCTS.

        Returns:
        - A newline-separated string of itac product names, or an error message if none are found.
        """
        logger.info("Tool called: list_itac_products")

        if not ITAC_API_TOKEN:
            logger.error("ITAC API token not configured. Please set ITAC_API_TOKEN in your environment.")
            return "ITAC API token not configured. Please set ITAC_API_TOKEN in your environment."

        if not ITAC_PRODUCTS:
            logger.error("ITAC products endpoint not configured. Please set ITAC_PRODUCTS in your environment.")
            return "ITAC products endpoint not configured. Please set ITAC_PRODUCTS in your environment."

        headers = {
            "Authorization": f"Bearer {ITAC_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "cloudaccountId": "702498770238",
            "productFilter": {
                "accountType": "ACCOUNT_TYPE_INTEL",
                "familyId": "61befbee-0607-47c5-b140-c4509dfef835",
                "region": "us-region-2"
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(ITAC_PRODUCTS, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                products = data.get("products")

                if not products:
                    logger.warning("No products found in response")
                    return "No products returned by the ITAC API."

                product_names = [item["name"] for item in products]
                output = "\n".join(product_names)
                logger.info(f"list_itac_products response:\n{output}")
                return output
        except Exception as e:
            logger.exception(f"Error fetching ITAC products: {e}")
            return "Failed to retrieve ITAC products."
