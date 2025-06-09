import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")
logger.info(f"INDEX_DIR: {INDEX_DIR}, EMBED_MODEL: {EMBED_MODEL}")

# Load vectorstore and retriever at module load
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
if not os.path.exists(INDEX_DIR):
    logger.error(f"FAISS index not found at {INDEX_DIR}. Please build the vectorstore first.")
    retriever = None
else:
    vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

def register_tools(mcp):
    @mcp.tool()
    async def document_qa(question: str) -> str:
        """
        Answer questions from internal documents using a RAG pipeline.
        """
        logger.info(f"Tool called: document_qa with question: {question}")
        if retriever is None:
            return "RAG vectorstore is not available."
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant information found in the documents."
        # Return the most relevant chunk(s)
        return "\n---\n".join([doc.page_content for doc in docs[:3]])