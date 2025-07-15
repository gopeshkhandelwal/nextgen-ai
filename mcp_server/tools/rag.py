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

class RerankedRetriever:
    """
    Enhanced retriever with reranking capability for improved relevance.
    """
    def __init__(self, base_retriever, reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.base_retriever = base_retriever
        self.reranker_model_name = reranker_model
        self.reranker = None
        self.reranker_enabled = os.getenv("RAG_ENABLE_RERANKER", "true").lower() == "true"
        
        if self.reranker_enabled:
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info("RerankedRetriever initialized with model: %s", reranker_model)
            except Exception as e:
                logger.warning("Failed to load reranker model: %s. Falling back to base retriever.", e)
                self.reranker_enabled = False
        else:
            logger.info("RerankedRetriever initialized without reranking (disabled or unavailable)")
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve and rerank documents for better relevance."""
        if not self.reranker_enabled:
            # Fallback to base retriever if reranker is unavailable
            return self.base_retriever.get_relevant_documents(query)
        
        # Get more candidates than needed for reranking
        candidate_multiplier = int(os.getenv("RAG_RERANK_CANDIDATE_MULTIPLIER", "3"))
        candidate_k = min(k * candidate_multiplier, 20)  # Cap at 20 to avoid excessive computation
        
        # Get initial candidates
        candidates = self.base_retriever.get_relevant_documents(query)
        
        if len(candidates) <= k:
            # If we have fewer candidates than requested, return all
            return candidates
        
        try:
            # Prepare query-document pairs for reranking
            pairs = [[query, doc.page_content] for doc in candidates]
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Combine documents with their scores
            scored_docs = list(zip(candidates, scores))
            
            # Sort by score (descending) and return top k
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
            logger.error("Error during reranking: %s. Falling back to base retriever.", e)
            return candidates[:k]
    
    async def get_relevant_documents_async(self, query: str, k: int = 5) -> List[Document]:
        """Async version of reranked retrieval."""
        if hasattr(self.base_retriever, 'get_relevant_documents_async'):
            candidates = await self.base_retriever.get_relevant_documents_async(query)
        else:
            # Fallback to sync version in executor
            loop = asyncio.get_event_loop()
            candidates = await loop.run_in_executor(
                None, 
                self.base_retriever.get_relevant_documents, 
                query
            )
        
        if not self.reranker_enabled or len(candidates) <= k:
            return candidates[:k]
        
        try:
            # Run reranking in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._rerank_documents,
                query,
                candidates,
                k
            )
        except Exception as e:
            logger.error("Error during async reranking: %s", e)
            return candidates[:k]
    
    def _rerank_documents(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        """Helper method for reranking documents."""
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]

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
        
        # Initialize reranked retriever
        self.reranked_retriever = RerankedRetriever(self.ensemble_retriever)
        
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
    
    def get_relevant_documents(self, query: str, use_reranking: bool = False):
        """Get relevant documents with caching and adaptive K."""
        # Check cache first
        cache_key = f"{query}_{self.base_k}_{'rerank' if use_reranking else 'base'}"
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        if query_hash in self._cache:
            logger.info("Cache hit for query: %s...", query[:50])
            return self._cache[query_hash]
        
        logger.info("Query not in cache: %s...", query[:50])
        
        # Get optimal k for this query
        optimal_k = self.get_optimal_k(query)
        
        # Choose retriever based on reranking preference
        if use_reranking:
            retriever = self.reranked_retriever
        else:
            retriever = self.ensemble_retriever
        
        # Temporarily adjust k values for base retrievers
        original_bm25_k = self.bm25_retriever.k
        original_vector_k = self.vector_retriever.search_kwargs.get("k", self.base_k)
        
        try:
            self.bm25_retriever.k = optimal_k
            self.vector_retriever.search_kwargs["k"] = optimal_k
            
            logger.info("Using retriever with k=%d, reranking=%s", optimal_k, use_reranking)
            
            if use_reranking:
                docs = retriever.get_relevant_documents(query, k=optimal_k)
            else:
                docs = retriever.get_relevant_documents(query)
            
            # Cache the result
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            logger.info("Adding query to cache: %s...", query[:50])
            self._cache[query_hash] = docs
            return docs
            
        finally:
            # Restore original k values
            self.bm25_retriever.k = original_bm25_k
            self.vector_retriever.search_kwargs["k"] = original_vector_k
    
    async def get_relevant_documents_async(self, query: str, use_reranking: bool = False):
        """Async version of document retrieval."""
        loop = asyncio.get_event_loop()
        
        if use_reranking:
            return await self.reranked_retriever.get_relevant_documents_async(query, k=self.get_optimal_k(query))
        else:
            return await loop.run_in_executor(
                self.executor, 
                self.get_relevant_documents, 
                query,
                use_reranking
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
    """Register optimized RAG tool with reranking support."""
    
    @mcp.tool()
    async def document_qa(question: str, search_method: str = "hybrid", use_reranking: bool = False) -> str:
        """
        Advanced document Q&A with optimized hybrid retrieval and optional reranking.
        
        Args:
            question: The question to answer
            search_method: Search method - "hybrid" (default), "semantic", "keyword", or "reranked"
            use_reranking: Whether to use reranking for better relevance (slower but more accurate)
        
        Features:
        - Adaptive document retrieval based on query complexity
        - Caching for improved performance
        - Multiple search strategies with fallback
        - Optional reranking for improved relevance
        - Comprehensive source attribution
        """
        logger.info("document_qa called: %s (method: %s, rerank: %s)", question[:100], search_method, use_reranking)
        
        if not hybrid_retriever:
            return "RAG system not available."
        
        try:
            # Handle reranked search method
            if search_method == "reranked":
                use_reranking = True
                search_method = "hybrid"  # Use hybrid as base for reranking
            
            # Choose retrieval method
            if search_method == "semantic":
                retriever = hybrid_retriever.vector_retriever
            elif search_method == "keyword":
                retriever = hybrid_retriever.bm25_retriever
            else:
                retriever = hybrid_retriever.ensemble_retriever
            
            # Get documents with optional reranking
            if use_reranking:
                docs = await hybrid_retriever.get_relevant_documents_async(question, use_reranking=True)
            else:
                docs = await hybrid_retriever.get_relevant_documents_async(question, use_reranking=False)
            
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
            rerank_info = " + reranked" if use_reranking else ""
            source_info = (
                f"\n\n📊 Sources: {len(docs)} documents "
                f"({search_method}{rerank_info} search, "
                f"K={hybrid_retriever.get_optimal_k(question)})"
            )
            
            logger.info("Answer generated: %s search, %d docs, reranked=%s", search_method, len(docs), use_reranking)
            return f"{answer.strip()}{source_info}"
            
        except Exception as e:
            logger.error("Error in document_qa: %s", e, exc_info=True)
            return f"Error processing query: {str(e)}"
