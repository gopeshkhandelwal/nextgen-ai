# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import sys
import logging
import traceback
from dotenv import load_dotenv
from fastmcp import FastMCP
from tools import weather, idc_images, idc_pools, rag, itac_products

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def register_all_tools(mcp_instance):
    """
    Register all tool modules to the MCP instance.

    Args:
        mcp_instance (FastMCP): An instance of the FastMCP server.
    """
    weather.register_tools(mcp_instance)
    idc_pools.register_tools(mcp_instance)
    idc_images.register_tools(mcp_instance)
    itac_products.register_tools(mcp_instance)
    rag.register_tools(mcp_instance)

def start_server():
    """
    Start the MCP server using stdio transport.
    """
    try:
        logger.info("🚀 Starting MCP server using stdio transport")
        mcp = FastMCP("mcp_server")
        register_all_tools(mcp)
        mcp.run(transport="stdio")
    except Exception as e:
        logger.critical("❌ Failed to start MCP server: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    start_server()