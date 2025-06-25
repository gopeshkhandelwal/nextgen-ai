import os
import sys
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from common_utils.config import get_llm

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_env_var(var_name):
    """
    Check if an environment variable is set, log and exit if not.
    """
    value = os.getenv(var_name)
    if not value:
        logger.critical("Environment variable '%s' is not set.", var_name)
        sys.exit(1)
    return value

def initialize_rag():
    """
    Initialize the RAG retriever and chain with robust error handling.

    Returns:
        tuple: (retriever, rag_chain)
    Exits the process if a critical error occurs.
    """
    try:
        index_dir = check_env_var("RAG_INDEX_DIR")
        embed_model = check_env_var("RAG_EMBED_MODEL")
        logger.info("RAG Vectorstore index: %s, Embedding model: %s", index_dir, embed_model)

        if not os.path.exists(embed_model):
            logger.error("Embedding model not found: %s. Please download the model.", embed_model)
            sys.exit(1)

        embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
        if not embedding_model:
            logger.critical("Failed to load embedding model: %s. Please download the model.", embed_model)
            sys.exit(1)

        if not os.path.exists(index_dir):
            logger.critical("FAISS index not found at %s. Please build the vectorstore.", index_dir)
            sys.exit(1)

        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        llm, _ = get_llm(tool_mode=False)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
        logger.info("FAISS vectorstore and RAG chain loaded successfully.")
        return retriever, rag_chain
    except Exception as e:
        logger.critical("Failed to initialize RAG pipeline: %s", e, exc_info=True)
        sys.exit(1)

# Initialize RAG components at startup
retriever, rag_chain = initialize_rag()

def register_tools(mcp):
    """
    Register RAG-based document QA tool with the MCP server.
    """
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
        logger.info("Tool called: document_qa with question: %s", question)
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
            logger.error("Error retrieving or synthesizing answer: %s", e, exc_info=True)
            return "Error retrieving or synthesizing answer from vectorstore."