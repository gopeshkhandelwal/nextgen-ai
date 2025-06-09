import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

if not INDEX_DIR or not EMBED_MODEL:
    logger.error("RAG_INDEX_DIR and RAG_EMBED_MODEL environment variables must be set.")
    retriever = None
else:
    logger.info(f"RAG Vectorstore index: {INDEX_DIR}, Embedding model: {EMBED_MODEL}")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    retriever = None
    if os.path.exists(INDEX_DIR):
        try:
            vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            logger.info("FAISS vectorstore loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS vectorstore: {e}")
    else:
        logger.warning(f"FAISS index not found at {INDEX_DIR}. Please build the vectorstore.")

def register_tools(mcp):
    @mcp.tool()
    async def document_qa(question: str) -> str:
        """
        Answer questions from internal documents using a RAG pipeline.
        Returns the most relevant document chunks or a message if not found.
        """
        logger.info(f"Tool called: document_qa with question: {question}")
        if retriever is None:
            logger.error("RAG vectorstore is not available.")
            return "RAG vectorstore is not available."
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return "Error retrieving documents from vectorstore."
        if not docs:
            logger.info("No relevant information found in the documents.")
            return "No relevant information found in the documents."
        logger.info(f"Returning {min(3, len(docs))} relevant document chunk(s).")
        return "\n---\n".join([doc.page_content for doc in docs[:3]])