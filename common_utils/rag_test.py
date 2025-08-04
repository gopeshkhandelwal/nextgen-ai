# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_DIR}. Run `build_vectorstore.py` first.")

# Create the embeddings object (fix: do NOT use a string here)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Load the FAISS vectorstore with the embeddings object
vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Now you can use retriever.invoke or retriever.get_relevant_documents
docs = retriever.invoke("ITAC gRPC API Guide")
print(docs)