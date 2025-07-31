# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import os
import sys
import logging
import asyncio
import hashlib
from typing import Optional, Dict, Any, List
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
from sentence_transformers import CrossEncoder

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

class HybridRAGRetriever:
    """
    Hybrid retriever with BM25 keyword search, semantic vector search,
    adaptive document selection, query caching, and optional reranking capabilities.
    
    Features:
    - Ensemble retrieval combining BM25 and semantic search
    - Adaptive K selection based on query complexity
    - LRU cache for improved performance
    - Optional cross-encoder reranking for enhanced relevance
    - Configurable search weights via environment variables
    - Async support for non-blocking operations
    """
    
    def __init__(self, documents, vectorstore, semantic_weight=None, keyword_weight=None):
        self.documents = documents
        self.vectorstore = vectorstore
        
        # Load weights from environment with defaults
        self.semantic_weight = float(semantic_weight or os.getenv("RAG_SEMANTIC_WEIGHT", "0.7"))
        self.keyword_weight = float(keyword_weight or os.getenv("RAG_KEYWORD_WEIGHT", "0.3"))
        self.base_k = int(os.getenv("RAG_RETRIEVAL_K", "5"))
        
        # Validate weights
        if abs(self.semantic_weight + self.keyword_weight - 1.0) > 0.0:
            logger.warning("Search weights don't sum to 1.0, normalizing...")
            total = self.semantic_weight + self.keyword_weight
            self.semantic_weight /= total
            self.keyword_weight /= total
        
        # Initialize cache
        self._cache = {}
        self._cache_size = int(os.getenv("RAG_CACHE_SIZE", "100"))
        
        # Thread executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize reranker
        self.reranker = None
        self.reranker_enabled = os.getenv("RAG_ENABLE_RERANKER", "false").lower()
        logger.info("reranker_enabled: %s", self.reranker_enabled)
        if self.reranker_enabled:
            try:
                reranker_model = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                self.reranker = CrossEncoder(reranker_model)
                logger.info("Reranker initialized with model: %s", reranker_model)
            except Exception as e:
                logger.warning("Failed to load reranker model: %s. Reranking disabled.", e)
                self.reranker_enabled = False
        
        # Initialize base retrievers
        self._initialize_retrievers()
        
        logger.info("HybridRAGRetriever initialized: BM25=%.2f, Semantic=%.2f, K=%d, Reranker=%s", 
                   self.keyword_weight, self.semantic_weight, self.base_k, self.reranker_enabled)
    
    def _initialize_retrievers(self):
        """Initialize BM25, vector, and ensemble retrievers."""
        # Create BM25 retriever for keyword search
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = self.base_k
        
        # Create vector retriever for semantic search
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.base_k}
        )
        
        # Create ensemble retriever combining both approaches
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.keyword_weight, self.semantic_weight]
        )
    
    def get_optimal_k(self, query: str) -> int:
        """
        Determine optimal number of documents to retrieve based on query complexity.
        
        Args:
            query: The input query string
            
        Returns:
            Optimal number of documents to retrieve
        """
        query_length = len(query.split())
        
        if query_length < 5:
            return max(2, self.base_k - 2)  # Simple queries need fewer docs
        elif query_length < 15:
            return self.base_k  # Medium complexity queries
        else:
            return min(self.base_k + 2, 10)  # Complex queries need more context
    
    def _rerank_documents(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        """
        Rerank documents using cross-encoder model for improved relevance.
        
        Args:
            query: The input query
            candidates: List of candidate documents
            k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not self.reranker_enabled or not self.reranker or len(candidates) <= k:
            return candidates[:k]
        
        try:
            # Prepare query-document pairs for reranking
            pairs = [[query, doc.page_content] for doc in candidates]
            
            # Get relevance scores from cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Combine documents with their scores and sort by relevance
            scored_docs = list(zip(candidates, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, score in scored_docs[:k]]
            
            # Log reranking statistics
            if len(candidates) > 0:
                avg_score = sum(scores) / len(scores)
                top_score = max(scores)
                logger.info("Reranked %d→%d docs, avg_score=%.3f, top_score=%.3f", 
                        len(candidates), len(reranked_docs), avg_score, top_score)
            
            return reranked_docs
            
        except Exception as e:
            logger.error("Error during reranking: %s. Using original order.", e)
            return candidates[:k]
    
    def _get_cache_key(self, query: str, use_reranking: bool) -> str:
        """Generate cache key for query."""
        return f"{query}_{self.base_k}_{'rerank' if use_reranking else 'base'}"
    
    def _manage_cache(self, query_hash: str, docs: List[Document]):
        """Manage LRU cache by adding new entry and removing old if needed."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[query_hash] = docs
    
    def get_relevant_documents(self, query: str, use_reranking: bool = False) -> List[Document]:
        """
        Retrieve relevant documents with caching, adaptive K selection, and optional reranking.
        
        Args:
            query: The input query string
            use_reranking: Whether to apply reranking for better relevance
            
        Returns:
            List of relevant documents
        """
        # Check cache first
        cache_key = self._get_cache_key(query, use_reranking)
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        if query_hash in self._cache:
            logger.info("Cache hit for query: %s...", query[:50])
            return self._cache[query_hash]
        
        logger.info("Cache miss for query: %s...", query[:50])
        
        # Get optimal k for this query
        optimal_k = self.get_optimal_k(query)
        
        # Temporarily adjust k values for base retrievers
        original_bm25_k = self.bm25_retriever.k
        original_vector_k = self.vector_retriever.search_kwargs.get("k", self.base_k)
        
        try:
            # If reranking, get more candidates for better selection
            if use_reranking and self.reranker_enabled:
                candidate_multiplier = int(os.getenv("RAG_RERANK_CANDIDATE_MULTIPLIER", "3"))
                retrieval_k = min(optimal_k * candidate_multiplier, 20)
            else:
                retrieval_k = optimal_k
            
            # Set temporary k values
            self.bm25_retriever.k = retrieval_k
            self.vector_retriever.search_kwargs["k"] = retrieval_k
            
            logger.info("Retrieving with k=%d, target=%d, reranking=%s", 
                    retrieval_k, optimal_k, use_reranking)
            
            # Get initial documents using ensemble retriever
            docs = self.ensemble_retriever.get_relevant_documents(query)
            
            # Apply reranking if requested and enabled
            if use_reranking and self.reranker_enabled and len(docs) > optimal_k:
                docs = self._rerank_documents(query, docs, optimal_k)
            elif len(docs) > optimal_k:
                docs = docs[:optimal_k]
            
            # Cache the result
            self._manage_cache(query_hash, docs)
            
            logger.info("Retrieved %d documents for query: %s...", len(docs), query[:50])
            return docs
            
        finally:
            # Always restore original k values
            self.bm25_retriever.k = original_bm25_k
            self.vector_retriever.search_kwargs["k"] = original_vector_k
    
    async def get_relevant_documents_async(self, query: str, use_reranking: bool = False) -> List[Document]:
        """
        Async version of document retrieval with optional reranking.
        
        Args:
            query: The input query string
            use_reranking: Whether to apply reranking for better relevance
            
        Returns:
            List of relevant documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_relevant_documents,
            query,
            use_reranking
        )
    
    def clear_cache(self):
        """Clear the query cache."""
        self._cache.clear()
        logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._cache_size,
            "cache_hit_ratio": "Not tracked"  # Could implement hit/miss tracking
        }

def create_enhanced_documents(vectorstore) -> List[Document]:
    """
    Create enhanced document chunks with improved metadata and chunking strategy.
    
    Args:
        vectorstore: The FAISS vectorstore instance
        
    Returns:
        List of processed documents with enhanced metadata
    """
    documents = []
    
    for doc_id, doc in vectorstore.docstore._dict.items():
        # Split large documents for better retrieval granularity
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
                        "chunk_size": len(chunk),
                        "is_chunked": True
                    }
                ))
        else:
            # Add metadata for non-chunked documents
            documents.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "original_doc_id": doc_id,
                    "chunk_size": len(doc.page_content),
                    "is_chunked": False
                }
            ))
    
    return documents

def check_env_var(var_name: str) -> str:
    """
    Check if an environment variable is set, log and exit if not.
    
    Args:
        var_name: Name of the environment variable to check
        
    Returns:
        The environment variable value
    """
    value = os.getenv(var_name)
    if not value:
        logger.critical("Environment variable '%s' is not set.", var_name)
        sys.exit(1)
    return value

def initialize_rag():
    """
    Initialize the production RAG system with hybrid retrieval capabilities.
    
    Returns:
        Tuple of (hybrid_retriever, rag_chain)
    """
    try:
        # Load required environment variables
        index_dir = check_env_var("RAG_INDEX_DIR")
        embed_model = check_env_var("RAG_EMBED_MODEL")
        
        logger.info("Initializing RAG system: index=%s, model=%s", index_dir, embed_model)

        # Validate embedding model path
        if not os.path.exists(embed_model):
            logger.error("Embedding model not found: %s. Please download the model.", embed_model)
            sys.exit(1)

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
        if not embedding_model:
            logger.critical("Failed to load embedding model: %s", embed_model)
            sys.exit(1)

        # Validate vectorstore index
        if not os.path.exists(index_dir):
            logger.critical("FAISS index not found at %s. Please build the vectorstore.", index_dir)
            sys.exit(1)

        # Load vectorstore
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
        
        # Create enhanced documents with better chunking
        documents = create_enhanced_documents(vectorstore)
        logger.info("Created %d enhanced document chunks", len(documents))        
        logger.debug("First 5 documents in loop: %s", documents[:5])
                
        
        # Initialize hybrid RAG retriever
        hybrid_retriever = HybridRAGRetriever(documents, vectorstore)        
        logger.info("RAG system initialized successfully")
        return hybrid_retriever
        
    except Exception as e:
        logger.critical("Failed to initialize RAG system: %s", e, exc_info=True)
        sys.exit(1)

# Initialize RAG components at module level
hybrid_retriever = initialize_rag()

def register_tools(mcp):
    """
    Register the document QA tool with the MCP server.
    
    Args:
        mcp: The MCP server instance
    """
    
    @mcp.tool()
    async def document_qa(question: str, search_method: str = "hybrid", use_reranking: bool = False) -> str:
        """
        Answer questions about IDC Compute gRPC APIs using advanced hybrid retrieval.
        
        This tool uses a combination of BM25 keyword search and semantic vector search
        to find relevant documents, with optional reranking for enhanced accuracy.
        
        Args:
            question: The question to answer
            search_method: Search method - "hybrid" (default), "semantic", or "keyword"
            use_reranking: Whether to use reranking for better relevance (slower but more accurate)
        
        Returns:
            A comprehensive answer with source attribution
            
        Features:
        - Adaptive document retrieval based on query complexity
        - Query caching for improved performance
        - Multiple search strategies (hybrid, semantic, keyword)
        - Optional cross-encoder reranking
        - Comprehensive source attribution
        
        Supported topics:
        - IDC Compute gRPC APIs and their specifications
        - Authentication methods including mTLS and Vault integration
        - Service operations (InstanceService, VNetService, etc.)
        - API testing with grpcurl
        - Protobuf definitions and Swagger documentation
        """
        logger.info("document_qa called: %s (method: %s, rerank: %s)", question[:100], search_method, use_reranking)
        
        if not hybrid_retriever:
            logger.error("RAG system not available")
            return "RAG system not available."
        
        try:
            # Validate search method
            valid_methods = ["hybrid", "semantic", "keyword"]
            if search_method not in valid_methods:
                logger.warning("Invalid search method: %s, using hybrid", search_method)
                search_method = "hybrid"
            
            # Get documents using the hybrid retriever
            if use_reranking:
                docs = await hybrid_retriever.get_relevant_documents_async(question, use_reranking=True)
            else:
                docs = await hybrid_retriever.get_relevant_documents_async(question, use_reranking=False)
            
            if not docs:
                logger.info("No relevant documents found for query: %s", question[:100])
                return "No relevant documents found for your question."
            
            # Select the appropriate base retriever for the RAG chain
            retriever_map = {
                "semantic": hybrid_retriever.vector_retriever,
                "keyword": hybrid_retriever.bm25_retriever,
                "hybrid": hybrid_retriever.ensemble_retriever
            }
            base_retriever = retriever_map[search_method]
            
            # Create RAG chain with selected retriever
            llm, _ = get_llm(tool_mode=False)
            temp_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=base_retriever,
                chain_type="stuff"
            )
            
            # Generate answer
            result = temp_chain.invoke({"query": question})
            answer = result.get("result", "") if isinstance(result, dict) else result
            
            if not answer.strip():
                logger.info("Empty answer generated for query: %s", question[:100])
                return "No relevant information found in the documents."
            
            # Create enhanced source information
            rerank_suffix = " + reranked" if use_reranking else ""
            optimal_k = hybrid_retriever.get_optimal_k(question)
            
            source_info = (
                f"\n\n📊 Sources: {len(docs)} documents "
                f"({search_method}{rerank_suffix} search, K={optimal_k})"
            )
            
            logger.info("Answer generated: %s search, %d docs, reranked=%s, chars=%d", 
                search_method, len(docs), use_reranking, len(answer))
            
            return f"{answer.strip()}{source_info}"
            
        except Exception as e:
            logger.error("Error in document_qa: %s", e, exc_info=True)
            return f"Error processing query: {str(e)}"
    
    @mcp.tool()
    async def rag_cache_stats() -> str:
        """Get RAG system cache statistics."""
        if not hybrid_retriever:
            return "RAG system not available."
        
        stats = hybrid_retriever.get_cache_stats()
        return f"Cache size: {stats['cache_size']}/{stats['max_cache_size']}"
    
    @mcp.tool()
    async def rag_clear_cache() -> str:
        """Clear the RAG system cache."""
        if not hybrid_retriever:
            return "RAG system not available."
        
        hybrid_retriever.clear_cache()
        return "RAG cache cleared successfully."
