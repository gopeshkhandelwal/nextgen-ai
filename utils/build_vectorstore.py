import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DOC_PATH = os.getenv("RAG_DOC_PATH")
INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

if not DOC_PATH or not INDEX_DIR or not EMBED_MODEL:
    logger.error("RAG_DOC_PATH, RAG_INDEX_DIR, and RAG_EMBED_MODEL must be set in .env")
    exit(1)

logger.info(f"DOC_PATH: {DOC_PATH}, INDEX_DIR: {INDEX_DIR}, EMBED_MODEL: {EMBED_MODEL}")
if not os.path.exists(EMBED_MODEL):
    logger.error(f"EMBED_MODEL not found: {EMBED_MODEL}. Please download the model.")
    exit(1)
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)

try:
    with open(DOC_PATH, "r") as f:
        content = f.read()
except Exception as e:
    logger.error(f"Failed to read document file: {e}")
    exit(1)

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(content)]
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local(INDEX_DIR)
logger.info(f"✅ FAISS index saved to {INDEX_DIR}")