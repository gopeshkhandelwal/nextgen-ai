import os
import sys
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from common_utils.config import get_llm

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines semantic search (FAISS) with keyword search (BM25) for better retrieval.
    """
    def __init__(self, documents, vectorstore, semantic_weight=0.7, keyword_weight=0.3):
        self.documents = documents
        self.vectorstore = vectorstore
        
        # Create BM25 retriever for keyword search
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 8
        
        # Create vector retriever for semantic search
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        
        # Combine both retrievers
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[keyword_weight, semantic_weight]  # 30% keyword, 70% semantic
        )
        
        logger.info("HybridRetriever initialized with weights: BM25=%.1f, Semantic=%.1f", 
                   keyword_weight, semantic_weight)
    
    def get_relevant_documents(self, query: str):
        """Get relevant documents using hybrid search."""
        return self.ensemble_retriever.get_relevant_documents(query)

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
    Initialize the RAG retriever and chain with hybrid search capability.

    Returns:
        tuple: (hybrid_retriever, rag_chain)
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

        # Load vectorstore
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        
        # Extract documents from vectorstore for BM25
        documents = []
        for doc_id, doc in vectorstore.docstore._dict.items():
            documents.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))
        
        logger.info("Loaded %d documents from vectorstore for hybrid retrieval", len(documents))
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(documents, vectorstore)
        
        # Create RAG chain with hybrid retriever
        llm, _ = get_llm(tool_mode=False)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever.ensemble_retriever,  # Use the ensemble retriever
            chain_type="stuff"
        )
        
        logger.info("FAISS vectorstore and Hybrid RAG chain loaded successfully.")
        return hybrid_retriever, rag_chain
        
    except Exception as e:
        logger.critical("Failed to initialize RAG pipeline: %s", e, exc_info=True)
        sys.exit(1)

# Initialize RAG components at startup
hybrid_retriever, rag_chain = initialize_rag()

def register_tools(mcp):
    """
    Register RAG-based document QA tool with the MCP server.
    """
    @mcp.tool()
    async def document_qa(question: str, search_method: str = "hybrid") -> str:
        """
        Answer questions about IDC Compute gRPC APIs, endpoints, authentication methods, Vault integration,
        and instance-related operations using retrieval-augmented generation (RAG).

        Args:
            question: The question to answer
            search_method: Search method - "hybrid" (default), "semantic", or "keyword"

        Returns a synthesized response using an LLM and the retrieved IDC gRPC API documentation.
        Always includes basic source information for transparency.

        Supported topics include:
        - Public/private IDC gRPC APIs and their Swagger or protobuf definitions
        - Authentication using mTLS and Vault annotations
        - Using grpcurl for testing or exploration
        - Service operations like InstanceService, VNetService, MachineImageService, etc.
        """
        logger.info("Tool called: document_qa with question: %s", question)
        
        if hybrid_retriever is None or rag_chain is None:
            logger.error("RAG vectorstore or chain is not available.")
            return "RAG vectorstore or chain is not available."
        
        try:
            # Choose retrieval method
            # default to hybrid if not specified
            retriever = hybrid_retriever.ensemble_retriever
            if search_method == "semantic":
                retriever = hybrid_retriever.vector_retriever
            elif search_method == "keyword":
                retriever = hybrid_retriever.bm25_retriever
 
            logger.info("Using %s retriever", retriever.__class__.__name__)
            # Create a temporary RAG chain with the chosen retriever
            llm, _ = get_llm(tool_mode=False)
            temp_rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            
            # Get the answer
            result = temp_rag_chain.invoke({"query": question})
            answer = result.get("result", "") if isinstance(result, dict) else result
            
            if not answer.strip():
                logger.info("No relevant information found in the documents.")
                return "No relevant information found in the documents."
            
            # Get document count for source info
            docs = hybrid_retriever.get_relevant_documents(question)
            docs_count = len(docs)
            
            # Add basic source information
            source_info = f"\n\n(Based on {docs_count} documents using {search_method} search)"
            
            logger.info("Answer generated using %s search method with %d documents", search_method, docs_count)
            return f"{answer.strip()}{source_info}"
            
        except Exception as e:
            logger.error("Error retrieving or synthesizing answer: %s", e, exc_info=True)
            return "Error retrieving or synthesizing answer from vectorstore."
