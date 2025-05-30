import sys
import traceback
import logging
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp_server.tools import weather, idc_images

load_dotenv()

logger = logging.getLogger("mcp_server")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

mcp = FastMCP("mcp_server")

# Register all tools
weather.register_tools(mcp)
idc_images.register_tools(mcp)

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting MCP server using stdio transport")
        mcp.run(transport="stdio")
    except Exception:
        logger.critical("Failed to start MCP server")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
