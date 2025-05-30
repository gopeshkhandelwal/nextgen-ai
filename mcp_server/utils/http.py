import httpx
import logging

logger = logging.getLogger("http_utils")

async def make_nws_request(url: str) -> dict | None:
    headers = {
        "User-Agent": "weather-mcp-agent/1.0",
        "Accept": "application/geo+json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
    except Exception:
        logger.exception("HTTP request failed")
        return None
