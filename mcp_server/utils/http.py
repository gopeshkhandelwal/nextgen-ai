# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import httpx
import logging

# Configure logger for this module
logger = logging.getLogger("http_utils")

async def make_nws_request(url: str) -> dict | None:
    """
    Make an asynchronous HTTP GET request to the given NWS (National Weather Service) API URL.

    Args:
        url (str): The API endpoint to query.

    Returns:
        dict | None: Parsed JSON response if successful, None otherwise.
    """
    headers = {
        "User-Agent": "weather-mcp-agent/1.0",
        "Accept": "application/geo+json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            logger.debug("NWS request to %s succeeded with status %d", url, response.status_code)
            return response.json()
    except Exception as e:
        logger.exception("HTTP request to %s failed: %s", url, e)
        return None
