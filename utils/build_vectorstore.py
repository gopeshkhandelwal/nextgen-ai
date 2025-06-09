import os
import warnings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DOC_PATH = os.getenv("RAG_DOC_PATH")
INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

logger.info(f"DOC_PATH: {DOC_PATH}, INDEX_DIR: {INDEX_DIR}, EMBED_MODEL {EMBED_MODEL}")
logger.info(f"INDEX_DIR: os.path.dirname(INDEX_DIR)")

os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)

with open(DOC_PATH, "r") as f:
    content = f.read()

docs = [Document(page_content=chunk) for chunk in CharacterTextSplitter(
    chunk_size=500, chunk_overlap=50).split_text(content)]
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index saved to {INDEX_DIR}")