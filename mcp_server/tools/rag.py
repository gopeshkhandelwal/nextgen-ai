import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from config import get_llm

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("RAG_INDEX_DIR")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL")

retriever = None
rag_chain = None

if not INDEX_DIR or not EMBED_MODEL:
    logger.error("RAG_INDEX_DIR and RAG_EMBED_MODEL environment variables must be set.")
else:
    logger.info(f"RAG Vectorstore index: {INDEX_DIR}, Embedding model: {EMBED_MODEL}")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if os.path.exists(INDEX_DIR):
        try:
            vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Retrieve more docs for LLM synthesis
            rag_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                retriever=retriever,
                chain_type="stuff"
            )
            logger.info("FAISS vectorstore and RAG chain loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS vectorstore: {e}")
    else:
        logger.warning(f"FAISS index not found at {INDEX_DIR}. Please build the vectorstore.")

def register_tools(mcp):
    @mcp.tool()
    async def document_qa(question: str) -> str:
        """
        Answer questions from internal documents using a RAG pipeline.
        Returns a synthesized answer using an LLM and retrieved documents.
        """
        logger.info(f"Tool called: document_qa with question: {question}")
        if retriever is None or rag_chain is None:
            logger.error("RAG vectorstore or chain is not available.")
            return "RAG vectorstore or chain is not available."
        try:
            # Industry best practice: Use the RAG chain to synthesize the answer from all retrieved docs
            result = rag_chain.invoke({"query": question})
            answer = result.get("result", "") if isinstance(result, dict) else result
            if not answer.strip():
                logger.info("No relevant information found in the documents.")
                return "No relevant information found in the documents."
            logger.info("Returning synthesized answer from RAG chain.")
            return answer.strip()
        except Exception as e:
            logger.error(f"Error retrieving or synthesizing answer: {e}")
            return "Error retrieving or synthesizing answer from vectorstore."