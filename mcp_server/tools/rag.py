import os
import sys
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from config import get_llm

logger = logging.getLogger(__name__)

def check_env_var(var_name):
    value = os.getenv(var_name)
    if not value:
        logger.critical(f"Environment variable '{var_name}' is not set.")
        sys.exit(1)
    return value

def initialize_rag():
    """
    Initialize the RAG retriever and chain with robust error handling.
    Returns:
        retriever, rag_chain
    Exits the process if a critical error occurs.
    """
    try:
        INDEX_DIR = check_env_var("RAG_INDEX_DIR")
        EMBED_MODEL = check_env_var("RAG_EMBED_MODEL")
        logger.info(f"RAG Vectorstore index: {INDEX_DIR}, Embedding model: {EMBED_MODEL}")
        if not os.path.exists(EMBED_MODEL):
            logger.error(f"EMBED_MODEL not found: {EMBED_MODEL}. Please download the model.")
            exit(1)
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        if not embedding_model:
            logger.critical(f"Failed to load embedding model: {EMBED_MODEL}. Please download the model.")
            sys.exit(1)
        if not os.path.exists(INDEX_DIR):
            logger.critical(f"FAISS index not found at {INDEX_DIR}. Please build the vectorstore.")
            sys.exit(1)
        vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        rag_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            retriever=retriever,
            chain_type="stuff"
        )
        logger.info("FAISS vectorstore and RAG chain loaded successfully.")
        return retriever, rag_chain
    except Exception as e:
        logger.critical(f"Failed to initialize RAG pipeline: {e}", exc_info=True)
        sys.exit(1)

# Initialize RAG components at startup
retriever, rag_chain = initialize_rag()

def register_tools(mcp):
    @mcp.tool()
    async def document_qa(question: str) -> str:
        """
        Answer questions about IDC Compute gRPC APIs, endpoints, authentication methods, Vault integration,
        and instance-related operations using retrieval-augmented generation (RAG).

        Returns a synthesized response using an LLM and the retrieved IDC gRPC API documentation.

        Supported topics include:
        - Public/private IDC gRPC APIs and their Swagger or protobuf definitions
        - Authentication using mTLS and Vault annotations
        - Using grpcurl for testing or exploration
        - Service operations like InstanceService, VNetService, MachineImageService, etc.
        """
        logger.info(f"Tool called: document_qa with question: {question}")
        if retriever is None or rag_chain is None:
            logger.error("RAG vectorstore or chain is not available.")
            return "RAG vectorstore or chain is not available."
        try:
            result = rag_chain.invoke({"query": question})
            answer = result.get("result", "") if isinstance(result, dict) else result
            if not answer.strip():
                logger.info("No relevant information found in the documents.")
                return "No relevant information found in the documents."
            logger.info("Returning synthesized answer from RAG chain.")
            return answer.strip()
        except Exception as e:
            logger.error(f"Error retrieving or synthesizing answer: {e}", exc_info=True)
            return "Error retrieving or synthesizing answer from vectorstore."