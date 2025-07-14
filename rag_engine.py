import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import pickle
import glob
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class RAGEngine:
    def __init__(self, vector_store_path: str = "vector_store", documents_path: str = "documents"):
        """Initialize the RAG engine with paths to vector store and documents"""
        self.vector_store_path = vector_store_path
        self.documents_path = documents_path
        self.embeddings_file = os.path.join(vector_store_path, "embeddings.pkl")
        self.document_chunks: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Create directories if they don't exist
        os.makedirs(vector_store_path, exist_ok=True)
        os.makedirs(documents_path, exist_ok=True)
        
        # Load existing embeddings if available
        self._load_embeddings()
        
        # Process all documents in the documents directory
        self._process_all_documents()
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
    
    def _load_embeddings(self) -> None:
        """Load embeddings from disk if they exist"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.document_chunks = data.get('chunks', {})
                    self.embeddings = data.get('embeddings', {})
                print(f"Loaded {len(self.embeddings)} embeddings from disk")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.document_chunks = {}
                self.embeddings = {}
    
    def _save_embeddings(self) -> None:
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump({
                'chunks': self.document_chunks,
                'embeddings': self.embeddings
            }, f)
        print(f"Saved {len(self.embeddings)} embeddings to disk")
    
    def _process_all_documents(self) -> None:
        """Process all documents in the documents directory"""
        document_files = glob.glob(os.path.join(self.documents_path, "*.txt"))
        for file_path in document_files:
            if os.path.basename(file_path) not in [chunk.split(':')[0] for chunk in self.document_chunks.keys()]:
                self.add_document(file_path)
    
    def _chunk_document(self, document_text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        for i in range(0, len(document_text), chunk_size - overlap):
            chunk = document_text[i:i + chunk_size]
            if len(chunk) > 0:  # Only add non-empty chunks
                chunks.append(chunk)
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using Gemini's embedding model"""
        embedding_model = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return np.array(embedding_model["embedding"])
    
    def add_document(self, file_path: str) -> None:
        """Process a document and add it to the vector store"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            # Get the document name from the file path
            document_name = os.path.basename(file_path)
            
            # Chunk the document
            chunks = self._chunk_document(document_text)
            
            # Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_name}:chunk_{i}"
                self.document_chunks[chunk_id] = chunk
                self.embeddings[chunk_id] = self._get_embedding(chunk)
            
            # Save embeddings to disk
            self._save_embeddings()
            
            print(f"Added document {document_name} with {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
    
    def _similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most relevant chunks for a query"""
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity scores
        similarities = {}
        for chunk_id, embedding in self.embeddings.items():
            similarities[chunk_id] = self._similarity(query_embedding, embedding)
        
        # Sort by similarity score
        sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k chunks
        top_chunks = []
        for chunk_id, score in sorted_chunks[:top_k]:
            top_chunks.append(self.document_chunks[chunk_id])
        
        return top_chunks
    
    def process_query(self, query: str) -> str:
        """Process a query using RAG"""
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "I don't have enough information to answer that question."
        
        # Combine chunks into context
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Create prompt with context and query
        prompt = f'''You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Answer the question based only on the provided context. If the context doesn't contain the information needed to answer that question, say "I don't have enough information to answer that question."'''
        
        # Generate response using Gemini
        response = self.model.generate_content(prompt)
        
        return response.text

# For compatibility with existing app.py
def create_vectorstore():
    """Initialize the RAG engine and process documents"""
    engine = RAGEngine()
    return engine

def query_context(query):
    """Query the RAG engine for context"""
    engine = RAGEngine()
    relevant_chunks = engine._retrieve_relevant_chunks(query)
    if not relevant_chunks:
        return "No relevant information found."
    return "\n\n---\n\n".join(relevant_chunks)
