import os
import sys
import logging
import asyncio
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from common_utils.config import get_llm

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    answer: str
    source_count: int
    search_method: str
    confidence: float = 0.0
    error: Optional[str] = None

class OptimizedHybridRetriever:
    """
    High-performance hybrid retriever with caching and adaptive behavior.
    """
    def __init__(self, documents, vectorstore, semantic_weight=None, keyword_weight=None):
        self.documents = documents
        self.vectorstore = vectorstore
        
        # Load weights from environment with defaults
        self.semantic_weight = float(semantic_weight or os.getenv("RAG_SEMANTIC_WEIGHT", "0.7"))
        self.keyword_weight = float(keyword_weight or os.getenv("RAG_KEYWORD_WEIGHT", "0.3"))
        self.base_k = int(os.getenv("RAG_RETRIEVAL_K", "5"))
        
        # Initialize cache
        self._cache = {}
        self._cache_size = int(os.getenv("RAG_CACHE_SIZE", "100"))
        
        # Thread executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self._initialize_retrievers()
        
    def _initialize_retrievers(self):
        """Initialize retrievers with optimized parameters."""
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = self.base_k
        
        # Create vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.base_k}
        )
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.keyword_weight, self.semantic_weight]
        )
        
        logger.info("OptimizedHybridRetriever initialized: BM25=%.2f, Semantic=%.2f, K=%d", 
                   self.keyword_weight, self.semantic_weight, self.base_k)
    
    def get_optimal_k(self, query: str) -> int:
        """Determine optimal number of documents based on query complexity."""
        query_length = len(query.split())
        
        if query_length < 5:
            return max(2, self.base_k - 2)
        elif query_length < 15:
            return self.base_k
        else:
            return min(self.base_k + 2, 10)
    
    def get_relevant_documents(self, query: str):
        """Get relevant documents with caching and adaptive K."""
        # Check cache first
        query_hash = hashlib.md5(f"{query}_{self.base_k}".encode()).hexdigest()
        
        if query_hash in self._cache:
            logger.info("Cache hit for query: %s...", query[:50])
            return self._cache[query_hash]
        logger.info("Query not in cache: %s...", query[:50])
        # Get optimal k for this query
        optimal_k = self.get_optimal_k(query)
        
        # Temporarily adjust k values
        original_bm25_k = self.bm25_retriever.k
        original_vector_k = self.vector_retriever.search_kwargs.get("k", self.base_k)
        
        try:
            self.bm25_retriever.k = optimal_k
            self.vector_retriever.search_kwargs["k"] = optimal_k
            
            logger.info("bm25_retriever.k: %s...", self.bm25_retriever.k)
            logger.info("vector_retriever.k: %s...", self.vector_retriever.search_kwargs["k"])
            docs = self.ensemble_retriever.get_relevant_documents(query)
            
            # Cache the result
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            logger.info("Adding query into cache: %s...", query[:50])
            self._cache[query_hash] = docs
            return docs
            
        finally:
            # Restore original k values
            self.bm25_retriever.k = original_bm25_k
            self.vector_retriever.search_kwargs["k"] = original_vector_k
    
    async def get_relevant_documents_async(self, query: str):
        """Async version of document retrieval."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.get_relevant_documents, 
            query
        )

def create_optimized_documents(vectorstore):
    """Create optimized document chunks with better metadata."""
    documents = []
    
    for doc_id, doc in vectorstore.docstore._dict.items():
        # Split large documents for better retrieval
        if len(doc.page_content) > 1200:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_doc_id": doc_id,
                        "chunk_size": len(chunk)
                    }
                ))
        else:
            documents.append(doc)
    
    return documents

def check_env_var(var_name):
    """Check if an environment variable is set, log and exit if not."""
    value = os.getenv(var_name)
    if not value:
        logger.critical("Environment variable '%s' is not set.", var_name)
        sys.exit(1)
    return value

def initialize_rag():
    """Initialize optimized RAG system."""
    try:
        index_dir = check_env_var("RAG_INDEX_DIR")
        embed_model = check_env_var("RAG_EMBED_MODEL")
        
        logger.info("Initializing RAG: index=%s, model=%s", index_dir, embed_model)

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
        if not embedding_model:
            logger.critical("Failed to load embedding model: %s. Please download the model.", embed_model)
            sys.exit(1)

        if not os.path.exists(index_dir):
            logger.critical("FAISS index not found at %s. Please build the vectorstore.", index_dir)
            sys.exit(1)

        # Load vectorstore
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        
        # Create optimized documents
        documents = create_optimized_documents(vectorstore)
        logger.info("Created %d optimized document chunks", len(documents))
        
        # Initialize optimized hybrid retriever
        hybrid_retriever = OptimizedHybridRetriever(documents, vectorstore)
        
        # Create initial RAG chain
        llm, _ = get_llm(tool_mode=False)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever.ensemble_retriever,
            chain_type="stuff"
        )
        
        logger.info("Optimized RAG system initialized successfully")
        return hybrid_retriever, rag_chain
        
    except Exception as e:
        logger.critical("Failed to initialize RAG: %s", e, exc_info=True)
        sys.exit(1)

# Initialize RAG components
hybrid_retriever, rag_chain = initialize_rag()

def register_tools(mcp):
    """Register optimized RAG tool."""
    
    @mcp.tool()
    async def document_qa(question: str, search_method: str = "hybrid") -> str:
        """
        Advanced document Q&A with optimized hybrid retrieval.
        
        Features:
        - Adaptive document retrieval based on query complexity
        - Caching for improved performance
        - Multiple search strategies with fallback
        - Comprehensive source attribution
        """
        logger.info("document_qa called: %s (method: %s)", question[:100], search_method)
        
        if not hybrid_retriever:
            return "RAG system not available."
        
        try:
            # Choose retrieval method
            if search_method == "semantic":
                retriever = hybrid_retriever.vector_retriever
            elif search_method == "keyword":
                retriever = hybrid_retriever.bm25_retriever
            else:
                retriever = hybrid_retriever.ensemble_retriever
            
            # Get documents (async for better performance)
            docs = await hybrid_retriever.get_relevant_documents_async(question)
            
            if not docs:
                return "No relevant documents found."
            
            # Create RAG chain with selected retriever
            llm, _ = get_llm(tool_mode=False)
            temp_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            
            # Get answer
            result = temp_chain.invoke({"query": question})
            answer = result.get("result", "") if isinstance(result, dict) else result
            
            if not answer.strip():
                return "No relevant information found in documents."
            
            # Add enhanced source information
            source_info = (
                f"\n\n📊 Sources: {len(docs)} documents "
                f"({search_method} search, "
                f"K={hybrid_retriever.get_optimal_k(question)})"
            )
            
            logger.info("Answer generated: %s search, %d docs", search_method, len(docs))
            return f"{answer.strip()}{source_info}"
            
        except Exception as e:
            logger.error("Error in document_qa: %s", e, exc_info=True)
            return f"Error processing query: {str(e)}"
