"""Optimized RAG Engine with Singleton Pattern
Handles document processing, embedding generation, and context retrieval
Includes advanced error handling, logging, and performance optimizations
"""

import os
import pickle
import glob
import threading
import time
import logging
import traceback
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_engine.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

class RAGEngine:
    """
    Singleton RAG Engine for efficient document retrieval and response generation.
    Handles embedding generation, document processing, and context retrieval.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("[INIT] Creating new RAGEngine instance")
                    cls._instance = super(RAGEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize RAG engine only once"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    print("[INIT] Initializing RAGEngine...")
                    self._setup_configuration()
                    self._setup_gemini()
                    self._setup_storage()
                    self._setup_cache()
                    self._load_embeddings()
                    self._process_documents()
                    RAGEngine._initialized = True
                    print("[INIT] RAGEngine initialization complete")
    
    def _setup_configuration(self):
        """Setup configuration from environment variables with validation"""
        try:
            self.documents_path = os.getenv("DOCUMENTS_PATH", "documents")
            self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "vector_store")
            
            # Parse and validate numeric configurations
            try:
                self.chunk_size = int(os.getenv("CHUNK_SIZE", "600"))
                if self.chunk_size <= 0:
                    logging.warning("Invalid CHUNK_SIZE, using default of 600")
                    self.chunk_size = 600
            except ValueError:
                logging.error("Invalid CHUNK_SIZE, using default of 600")
                self.chunk_size = 600
                
            try:
                self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
                if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
                    logging.warning("Invalid CHUNK_OVERLAP, using default of 100")
                    self.chunk_overlap = 100
            except ValueError:
                logging.error("Invalid CHUNK_OVERLAP, using default of 100")
                self.chunk_overlap = 100
                
            try:
                self.max_context_chunks = int(os.getenv("MAX_CONTEXT_CHUNKS", "2"))
                if self.max_context_chunks <= 0:
                    logging.warning("Invalid MAX_CONTEXT_CHUNKS, using default of 2")
                    self.max_context_chunks = 2
            except ValueError:
                logging.error("Invalid MAX_CONTEXT_CHUNKS, using default of 2")
                self.max_context_chunks = 2
            
            self.embeddings_file = os.path.join(self.vector_store_path, "embeddings.pkl")
            
            # Cache configuration with validation
            try:
                self.cache_size = int(os.getenv("CACHE_SIZE", "100"))
                if self.cache_size <= 0:
                    logging.warning("Invalid CACHE_SIZE, using default of 100")
                    self.cache_size = 100
            except ValueError:
                logging.error("Invalid CACHE_SIZE, using default of 100")
                self.cache_size = 100
                
            try:
                self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # Time to live in seconds (1 hour default)
                if self.cache_ttl <= 0:
                    logging.warning("Invalid CACHE_TTL, using default of 3600")
                    self.cache_ttl = 3600
            except ValueError:
                logging.error("Invalid CACHE_TTL, using default of 3600")
                self.cache_ttl = 3600
            
            # Validate API key
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            
            logging.info(f"Configuration: chunks={self.chunk_size}, overlap={self.chunk_overlap}, "
                        f"max_context={self.max_context_chunks}, cache_size={self.cache_size}")
                        
        except Exception as e:
            logging.critical(f"Failed to setup configuration: {e}")
            logging.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to setup configuration: {e}")
    
    def _setup_gemini(self):
        """Initialize Gemini API client"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("[INIT] Gemini API client initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini API: {e}")
            raise
    
    def _setup_storage(self):
        """Setup storage directories and data structures"""
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.documents_path, exist_ok=True)
        
        self.document_chunks: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        print(f"[INIT] Storage directories created: {self.vector_store_path}, {self.documents_path}")
        
    def _setup_cache(self):
        """Setup caching for frequently accessed embeddings and query results"""
        # Cache for query embeddings to avoid regenerating for similar queries
        self.query_embedding_cache: Dict[str, Tuple[np.ndarray, float]] = {}  # (query, (embedding, timestamp))
        
        # Cache for query results to avoid recomputing for identical queries
        self.query_result_cache: Dict[str, Tuple[str, float]] = {}  # (query, (context, timestamp))
        
        # Cache for frequently accessed chunk embeddings
        self.chunk_access_count: Dict[str, int] = {}  # Track access frequency
        
        print(f"[INIT] Cache initialized with size={self.cache_size}, ttl={self.cache_ttl}s")
    
    def _load_embeddings(self):
        """Load existing embeddings from disk with robust error handling"""
        if os.path.exists(self.embeddings_file):
            try:
                start_time = time.time()
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Validate loaded data structure
                if not isinstance(data, dict):
                    raise ValueError("Invalid embeddings file format: root object is not a dictionary")
                    
                if 'chunks' not in data or 'embeddings' not in data:
                    raise ValueError("Invalid embeddings file format: missing required keys")
                    
                self.document_chunks = data.get('chunks', {})
                self.embeddings = data.get('embeddings', {})
                
                # Validate that chunks and embeddings match
                if set(self.document_chunks.keys()) != set(self.embeddings.keys()):
                    logging.warning("Mismatch between chunks and embeddings keys, fixing...")
                    # Keep only keys that exist in both dictionaries
                    common_keys = set(self.document_chunks.keys()) & set(self.embeddings.keys())
                    self.document_chunks = {k: self.document_chunks[k] for k in common_keys}
                    self.embeddings = {k: self.embeddings[k] for k in common_keys}
                
                load_time = time.time() - start_time
                logging.info(f"Loaded {len(self.embeddings)} embeddings from disk in {load_time:.2f}s")
                
            except (pickle.PickleError, EOFError) as e:
                logging.error(f"Failed to load embeddings due to file corruption: {e}")
                logging.info("Creating backup of corrupted embeddings file")
                
                # Create backup of corrupted file
                backup_path = f"{self.embeddings_file}.bak.{int(time.time())}"
                try:
                    os.rename(self.embeddings_file, backup_path)
                    logging.info(f"Corrupted embeddings file backed up to {backup_path}")
                except Exception as backup_error:
                    logging.error(f"Failed to backup corrupted embeddings file: {backup_error}")
                
                logging.info("Starting with empty embeddings")
                self.document_chunks = {}
                self.embeddings = {}
                
            except Exception as e:
                logging.error(f"Failed to load embeddings: {e}")
                logging.debug(traceback.format_exc())
                logging.info("Starting with empty embeddings")
                self.document_chunks = {}
                self.embeddings = {}
        else:
            logging.info("No existing embeddings found, starting fresh")
    
    def _save_embeddings(self):
        """Save embeddings to disk with robust error handling and atomic writes"""
        try:
            start_time = time.time()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            
            # Write to temporary file first (atomic write pattern)
            temp_file = f"{self.embeddings_file}.tmp"
            
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump({
                        'chunks': self.document_chunks,
                        'embeddings': self.embeddings,
                        'metadata': {
                            'timestamp': time.time(),
                            'version': '1.1',
                            'chunk_count': len(self.document_chunks),
                            'embedding_count': len(self.embeddings)
                        }
                    }, f)
                
                # Create backup of current file if it exists
                if os.path.exists(self.embeddings_file):
                    backup_file = f"{self.embeddings_file}.bak"
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                        os.rename(self.embeddings_file, backup_file)
                    except Exception as backup_error:
                        logging.warning(f"Failed to create backup during save: {backup_error}")
                
                # Rename temp file to actual file (atomic operation)
                os.rename(temp_file, self.embeddings_file)
                
                save_time = time.time() - start_time
                logging.info(f"Saved {len(self.embeddings)} embeddings to disk in {save_time:.2f}s")
                
            except Exception as write_error:
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise write_error
                
        except Exception as e:
            logging.error(f"Failed to save embeddings: {e}")
            logging.debug(traceback.format_exc())
            # Don't re-raise to allow operation to continue
    
    def _process_documents(self):
        """Process all documents in the documents directory"""
        try:
            document_files = glob.glob(os.path.join(self.documents_path, "*.txt"))
            print(f"[INIT] Found {len(document_files)} document files")
            
            for file_path in document_files:
                filename = os.path.basename(file_path)
                # Check if document is already processed
                existing_chunks = [chunk_id for chunk_id in self.document_chunks.keys() 
                                 if chunk_id.startswith(f"{filename}:")]
                
                if not existing_chunks:
                    print(f"[INIT] Processing new document: {filename}")
                    self.add_document(file_path)
                else:
                    print(f"[INIT] Document already processed: {filename} ({len(existing_chunks)} chunks)")
                    
        except Exception as e:
            print(f"[ERROR] Error processing documents: {e}")
    
    def _chunk_document(self, document_text: str) -> List[str]:
        """Split document into overlapping chunks with optimized size and semantic boundaries"""
        if not document_text.strip():
            return []
        
        # Split text into paragraphs first for more semantic chunking
        paragraphs = [p.strip() for p in document_text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk and start new one
            if current_size + len(paragraph) > self.chunk_size:
                if current_chunk:  # Only add non-empty chunks
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_size = len(paragraph)
            else:
                # Add paragraph to current chunk with a separator
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += len(paragraph)
        
        # Add the last chunk if it exists
        if current_chunk and len(current_chunk) > 50:  # Only add meaningful chunks
            chunks.append(current_chunk.strip())
        
        # If chunks are too large or too few, fall back to the original method
        if not chunks or any(len(chunk) > self.chunk_size * 1.5 for chunk in chunks):
            chunks = []
            for i in range(0, len(document_text), self.chunk_size - self.chunk_overlap):
                chunk = document_text[i:i + self.chunk_size].strip()
                if len(chunk) > 50:  # Only add meaningful chunks
                    chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text with caching and error handling"""
        try:
            start_time = time.time()
            
            # Check if this is a query and if it's in the cache
            if len(text) < 1000:  # Assume it's a query if it's short
                # Normalize text for cache key
                cache_key = text.lower().strip()
                
                # Check cache
                if cache_key in self.query_embedding_cache:
                    embedding, timestamp = self.query_embedding_cache[cache_key]
                    # Check if cache entry is still valid
                    if time.time() - timestamp < self.cache_ttl:
                        logging.debug(f"[CACHE] Query embedding cache hit for: {cache_key[:30]}...")
                        return embedding
                    else:
                        # Remove expired cache entry
                        del self.query_embedding_cache[cache_key]
            
            # Generate new embedding with retry logic
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model="models/gemini-embedding-2",
                        content=text,
                        task_type="retrieval_query"
                    )
                    embedding = np.array(result["embedding"])
                    
                    # Validate embedding
                    if embedding.size == 0:
                        raise ValueError("Empty embedding returned from API")
                    
                    # Cache the result if it's a query
                    if len(text) < 1000:
                        # Manage cache size
                        if len(self.query_embedding_cache) >= self.cache_size:
                            # Remove oldest entry
                            oldest_key = min(self.query_embedding_cache.keys(), 
                                            key=lambda k: self.query_embedding_cache[k][1])
                            del self.query_embedding_cache[oldest_key]
                        
                        # Add to cache
                        cache_key = text.lower().strip()
                        self.query_embedding_cache[cache_key] = (embedding, time.time())
                        logging.debug(f"[CACHE] Cached query embedding for: {cache_key[:30]}...")
                    
                    # Log performance metrics
                    embedding_time = time.time() - start_time
                    logging.debug(f"Generated embedding in {embedding_time:.3f}s")
                    
                    return embedding
                    
                except Exception as retry_error:
                    if attempt < max_retries - 1:
                        logging.warning(f"Embedding generation attempt {attempt+1} failed: {retry_error}. Retrying...")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise retry_error
                        
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            logging.debug(traceback.format_exc())
            return None
    
    def add_document(self, file_path: str) -> bool:
        """Add a document to the knowledge base"""
        try:
            print(f"[INIT] Adding document: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            if not document_text.strip():
                print(f"[ERROR] Document is empty: {file_path}")
                return False
            
            filename = os.path.basename(file_path)
            chunks = self._chunk_document(document_text)
            
            if not chunks:
                print(f"[ERROR] No valid chunks created from: {file_path}")
                return False
            
            # Generate embeddings for chunks
            successful_chunks = 0
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}:chunk_{i}"
                embedding = self._get_embedding(chunk)
                
                if embedding is not None:
                    self.document_chunks[chunk_id] = chunk
                    self.embeddings[chunk_id] = embedding
                    successful_chunks += 1
                else:
                    print(f"[ERROR] Failed to create embedding for chunk {i} in {filename}")
            
            if successful_chunks > 0:
                self._save_embeddings()
                print(f"[INIT] Successfully added {successful_chunks}/{len(chunks)} chunks from {filename}")
                return True
            else:
                print(f"[ERROR] No chunks successfully processed from {filename}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error adding document {file_path}: {e}")
            return False
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"[ERROR] Error calculating similarity: {e}")
            return 0.0
            
    def _expand_query(self, query: str) -> str:
        """Expand query to improve semantic matching"""
        try:
            # Simple keyword extraction and expansion
            # In a production system, this could use more sophisticated NLP techniques
            keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            # Add synonyms or related terms for insurance domain
            domain_terms = {
                "premium": "payment fee cost",
                "policy": "plan coverage insurance",
                "claim": "reimbursement payout benefit",
                "coverage": "protection benefit insurance",
                "term": "period duration time",
                "fund": "investment money asset",
                "return": "yield profit gain",
                "withdraw": "redeem cash take"
            }
            
            expanded_terms = []
            for keyword in keywords:
                expanded_terms.append(keyword)
                for term, synonyms in domain_terms.items():
                    if keyword in term or term in keyword:
                        expanded_terms.extend(synonyms.split())
            
            # Combine original query with expanded terms
            expanded_query = query + " " + " ".join(set(expanded_terms))
            return expanded_query
            
        except Exception as e:
            print(f"[ERROR] Error expanding query: {e}")
            return query  # Fall back to original query
            
    def _apply_mmr(self, similarities: List[Tuple[str, float, np.ndarray]], query_embedding: np.ndarray, 
                   lambda_param: float = 0.5, max_chunks: int = None) -> List[Tuple[str, float]]:
        """Apply Maximal Marginal Relevance to select diverse yet relevant chunks"""
        if not max_chunks:
            max_chunks = self.max_context_chunks
            
        if not similarities:
            return []
            
        # Initialize with the most similar chunk
        selected_ids = [similarities[0][0]]
        selected_similarities = [(similarities[0][0], similarities[0][1])]
        
        # Store remaining candidates
        candidates = similarities[1:]
        
        # Select remaining chunks using MMR
        while len(selected_ids) < max_chunks and candidates:
            max_mmr = -1
            max_idx = -1
            
            for i, (chunk_id, similarity, embedding) in enumerate(candidates):
                # Calculate the maximum similarity to any already selected chunk
                max_sim_to_selected = 0
                for selected_id in selected_ids:
                    # Find the embedding for the selected chunk
                    for c_id, _, c_embedding in similarities:
                        if c_id == selected_id:
                            sim = self._calculate_similarity(embedding, c_embedding)
                            max_sim_to_selected = max(max_sim_to_selected, sim)
                            break
                
                # Calculate MMR score: balance between relevance and diversity
                mmr = lambda_param * similarity - (1 - lambda_param) * max_sim_to_selected
                
                if mmr > max_mmr:
                    max_mmr = mmr
                    max_idx = i
            
            if max_idx == -1:
                break
                
            # Add the chunk with highest MMR score
            selected_ids.append(candidates[max_idx][0])
            selected_similarities.append((candidates[max_idx][0], candidates[max_idx][1]))
            candidates.pop(max_idx)
        
        return selected_similarities
    
    def get_context(self, query: str) -> str:
        """Retrieve relevant context for a query using semantic search with query expansion and caching"""
        try:
            start_time = time.time()
            logging.info(f"Retrieving context for query: {query[:50]}...")
            
            # Check result cache first
            cache_key = query.lower().strip()
            if cache_key in self.query_result_cache:
                context, timestamp = self.query_result_cache[cache_key]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.cache_ttl:
                    logging.info(f"Query result cache hit for: {cache_key[:30]}...")
                    return context
                else:
                    # Remove expired cache entry
                    logging.debug(f"Query result cache expired for: {cache_key[:30]}...")
                    del self.query_result_cache[cache_key]
            
            # Validate query
            if not query or len(query.strip()) == 0:
                logging.warning("Empty query received")
                return "Please provide a valid query."
            
            if not self.embeddings:
                logging.warning("No embeddings available")
                return "No relevant information found."
            
            # Query expansion for better semantic matching
            expanded_query = self._expand_query(query)
            logging.debug(f"Expanded query: {expanded_query[:50]}...")
            
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self._get_embedding(expanded_query)
            embedding_time = time.time() - embedding_start
            
            if query_embedding is None:
                logging.error("Failed to generate query embedding")
                return "Unable to process query."
            
            logging.debug(f"Generated query embedding in {embedding_time:.3f}s")
            
            # Calculate similarities with MMR (Maximal Marginal Relevance)
            # to balance relevance and diversity
            similarity_start = time.time()
            similarities = []
            for chunk_id, embedding in self.embeddings.items():
                # Update access count for this chunk
                self.chunk_access_count[chunk_id] = self.chunk_access_count.get(chunk_id, 0) + 1
                
                similarity = self._calculate_similarity(query_embedding, embedding)
                similarities.append((chunk_id, similarity, embedding))
            
            similarity_time = time.time() - similarity_start
            logging.debug(f"Calculated {len(similarities)} similarities in {similarity_time:.3f}s")
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply MMR to select diverse yet relevant chunks
            mmr_start = time.time()
            selected_chunks = self._apply_mmr(similarities, query_embedding)
            mmr_time = time.time() - mmr_start
            logging.debug(f"Applied MMR in {mmr_time:.3f}s, selected {len(selected_chunks)} chunks")
            
            if not selected_chunks or selected_chunks[0][1] < 0.1:  # Very low similarity threshold
                logging.warning("No relevant context found")
                return "No relevant information found."
            
            # Combine selected chunks
            context_parts = []
            for chunk_id, similarity in selected_chunks:
                chunk_content = self.document_chunks.get(chunk_id, "")
                if chunk_content:
                    context_parts.append(chunk_content)
                    logging.debug(f"Using chunk {chunk_id} (similarity: {similarity:.3f})")
                else:
                    logging.warning(f"Chunk ID {chunk_id} found in embeddings but not in document_chunks")
            
            context = "\n\n---\n\n".join(context_parts)
            logging.info(f"Retrieved {len(context_parts)} chunks, total length: {len(context)}")
            
            # Cache the result
            if len(self.query_result_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.query_result_cache.keys(), 
                                key=lambda k: self.query_result_cache[k][1])
                del self.query_result_cache[oldest_key]
                logging.debug(f"Removed oldest cache entry: {oldest_key[:30]}...")
            
            self.query_result_cache[cache_key] = (context, time.time())
            logging.debug(f"Cached query result for: {cache_key[:30]}...")
            
            # Log performance metrics
            total_time = time.time() - start_time
            logging.info(f"Retrieved context for query in {total_time:.3f}s (embedding: {embedding_time:.3f}s, similarity: {similarity_time:.3f}s, mmr: {mmr_time:.3f}s)")
            
            return context
            
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            logging.debug(traceback.format_exc())
            return "Unable to retrieve relevant information."
    
    def query(self, user_input: str, temperature: float = 0.2) -> str:
        """Process a complete query with context retrieval and response generation with robust error handling"""
        start_time = time.time()
        logging.info(f"Processing query: {user_input[:50]}...")
        
        try:
            # Validate input
            if not user_input or len(user_input.strip()) == 0:
                logging.warning("Empty query received")
                return "Please provide a valid question."
                
            if temperature < 0.0 or temperature > 1.0:
                logging.warning(f"Invalid temperature value: {temperature}, using default 0.2")
                temperature = 0.2
            
            # Get relevant context with timing
            context_start = time.time()
            context = self.get_context(user_input)
            context_time = time.time() - context_start
            logging.debug(f"Retrieved context in {context_time:.3f}s, length: {len(context)}")
            
            # Create optimized prompt
            prompt = f"""You are Veena, insurance advisor for ValuEnable Life Insurance.

Context: {context}

User: {user_input}

Respond under 20 words, always end with a question."""
            
            # Send request to Gemini API with retry logic
            max_retries = 2
            retry_delay = 1  # seconds
            api_time = 0
            
            for attempt in range(max_retries + 1):
                try:
                    api_start = time.time()
                    print("[GEMINI] Sending request to Gemini...")
                    response = self.model.generate_content(prompt)
                    api_time = time.time() - api_start
                    logging.debug(f"Gemini API response received in {api_time:.3f}s")
                    break
                    
                except Exception as api_error:
                    if attempt < max_retries:
                        logging.warning(f"API request attempt {attempt+1} failed: {api_error}. Retrying...")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise api_error
            
            if response and response.text:
                reply = response.text.strip()
                # Strip markdown for TTS compatibility
                reply = reply.replace("*", "")
                
                # Fallback for empty responses
                if not reply:
                    reply = "I'm sorry, could you please repeat your question?"
                
                # Log performance metrics
                total_time = time.time() - start_time
                logging.info(f"Query processed in {total_time:.3f}s (context: {context_time:.3f}s, api: {api_time:.3f}s)")
                
                print(f"[RESPONSE] Generated response: {reply}")
                return reply
            else:
                logging.error("Invalid or empty response from Gemini API")
                print("[ERROR] Empty response from Gemini")
                return "I'm sorry, could you please repeat your question?"
                
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            logging.debug(traceback.format_exc())
            print(f"[ERROR] Error processing query: {e}")
            return "I'm having trouble right now. How can I help you with your policy?"

# Convenience functions for backward compatibility
def create_vectorstore():
    """Initialize the RAG engine (backward compatibility)"""
    print("[INIT] Initializing RAG engine...")
    RAGEngine()
    print("[INIT] RAG engine ready")

def query_context(query: str) -> str:
    """Get context for a query (backward compatibility)"""
    engine = RAGEngine()
    return engine.get_context(query)