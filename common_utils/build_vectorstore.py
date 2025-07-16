import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_env_and_validate():
    """Load and validate required environment variables."""
    load_dotenv()
    doc_path = os.getenv("RAG_DOC_PATH")
    index_dir = os.getenv("RAG_INDEX_DIR")
    embed_model = os.getenv("RAG_EMBED_MODEL")

    if not doc_path or not index_dir or not embed_model:
        logger.error("Environment variables RAG_DOC_PATH, RAG_INDEX_DIR, and RAG_EMBED_MODEL are required.")
        raise EnvironmentError("Missing required environment variables.")

    if not os.path.exists(embed_model):
        logger.error("Embedding model path does not exist: %s", embed_model)
        raise FileNotFoundError(f"Model not found at {embed_model}")

    os.makedirs(os.path.dirname(index_dir), exist_ok=True)

    logger.info("Environment loaded successfully.")
    logger.info("Document Path: %s", doc_path)
    logger.info("Index Directory: %s", index_dir)
    logger.info("Embedding Model: %s", embed_model)

    return doc_path, index_dir, embed_model

def read_document(file_path):
    """Read content from a document file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        logger.exception("Failed to read document file.")
        raise e

def build_and_save_index(doc_path, index_dir, embed_model):
    """Build FAISS index from document and save to disk."""
    content = read_document(doc_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(content)]
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(index_dir)
    logger.info("✅ FAISS index saved to %s", index_dir)

def main():
    try:
        doc_path, index_dir, embed_model = load_env_and_validate()
        build_and_save_index(doc_path, index_dir, embed_model)
    except Exception as err:
        logger.critical("Indexing failed: %s", err)

if __name__ == "__main__":
    main()