# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import argparse
import logging
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name: str, output_dir: str):
    """
    Download a Hugging Face model and save it to a specified directory.

    Args:
        model_name (str): The Hugging Face model repo ID.
        output_dir (str): Local directory to store the downloaded model.
    """
    logger.info("Starting download: %s --> %s", model_name, output_dir)
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN")
        )
        logger.info("✅ Model downloaded successfully to %s", output_dir)
    except Exception as e:
        logger.exception("❌ Failed to download model: %s", e)

def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("--model", required=True, help="Hugging Face model repo ID (e.g., bert-base-uncased)")
    parser.add_argument("--output_dir", required=True, help="Target directory for storing the model")
    args = parser.parse_args()
    download_model(args.model, args.output_dir)

if __name__ == "__main__":
    main()