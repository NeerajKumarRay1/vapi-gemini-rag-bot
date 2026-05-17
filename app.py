"""
Optimized Flask Application for RAG Chatbot
Provides clean, robust API endpoints with centralized error handling
"""

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import requests
import time
from typing import Dict, Any, Optional
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("[ERROR] GEMINI_API_KEY environment variable is required")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

class ErrorHandler:
    """Centralized error handling for consistent responses"""
    
    @staticmethod
    def handle_api_error(error: Exception) -> str:
        """Handle Gemini API errors"""
        print(f"[ERROR] API Error: {error}")
        return "I'm having trouble connecting right now. What can I help you with?"
    
    @staticmethod
    def handle_processing_error(error: Exception) -> str:
        """Handle processing errors"""
        print(f"[ERROR] Processing Error: {error}")
        return "I'm having trouble processing that. Could you try again?"
    
    @staticmethod
    def get_fallback_response() -> str:
        """Get fallback response for empty replies"""
        return "I'm sorry, could you please repeat your question?"

def validate_input(data: Dict[str, Any]) -> Optional[str]:
    """Validate and extract user input from request data"""
    try:
        # Handle OpenAI-style messages format
        if "messages" in data and len(data["messages"]) > 0:
            return data["messages"][-1]["content"].strip()
        
        # Handle Vapi transcript format
        if "transcript" in data:
            return data.get("transcript", "").strip()
        
        # No valid input found
        return None
        
    except Exception as e:
        print(f"[ERROR] Input validation failed: {e}")
        return None

def create_json_response(text: str) -> Dict[str, str]:
    """Create consistent JSON response format"""
    response = {"text": text}
    print(f"🚀 JSON sent to client: {response}")
    return response

@app.route("/")
def index():
    """Serve the chat interface"""
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"[ERROR] Failed to serve index: {e}")
        return "Chat interface temporarily unavailable", 500

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint with RAG integration"""
    start_time = time.time()
    
    try:
        # Log incoming request
        data = request.get_json()
        print(f"💡 Received request: {data}")
        
        # Validate input
        user_input = validate_input(data)
        if not user_input:
            print("[ERROR] No valid input found in request")
            return jsonify(create_json_response(ErrorHandler.get_fallback_response()))
        
        print(f"📝 User input: {user_input}")
        
        # Get RAG engine instance
        rag_engine = RAGEngine()
        
        # Get context from RAG engine
        context = rag_engine.get_context(user_input)
        print(f"📚 Context retrieved: {len(context)} characters")
        
        # Create optimized prompt
        prompt = f"""You are Veena, insurance advisor for ValuEnable Life Insurance.

Context: {context}

User: {user_input}

Respond under 20 words, always end with a question."""
        
        # Call Gemini API
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
        
        print("[GEMINI] Sending request to Gemini API...")
        response = requests.post(
            GEMINI_URL, 
            headers={"Content-Type": "application/json"}, 
            json=payload,
            timeout=10  # 10 second timeout
        )
        
        print(f"🌐 Gemini response status: {response.status_code}")
        
        # Process Gemini response
        if response.status_code == 200:
            response_data = response.json()
            
            if "candidates" in response_data and response_data["candidates"]:
                reply = response_data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                reply = "I'm sorry, I couldn't generate a response right now."
        else:
            print(f"[ERROR] Gemini API error: {response.status_code} - {response.text}")
            reply = ErrorHandler.handle_api_error(Exception(f"API returned {response.status_code}"))
        
        # Clean up response
        if not reply.strip():
            reply = ErrorHandler.get_fallback_response()
        
        # Strip markdown for TTS compatibility
        reply = reply.replace("*", "").replace("**", "").replace("_", "")
        
        # Log performance
        processing_time = time.time() - start_time
        print(f"[PERF] Request processed in {processing_time:.2f} seconds")
        print(f"✅ Final response: {reply}")
        
        return jsonify(create_json_response(reply))
        
    except requests.exceptions.Timeout:
        print("[ERROR] Gemini API timeout")
        return jsonify(create_json_response("I'm taking too long to respond. What's your question?"))
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error: {e}")
        return jsonify(create_json_response(ErrorHandler.handle_api_error(e)))
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in chat endpoint: {e}")
        return jsonify(create_json_response(ErrorHandler.handle_processing_error(e)))

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    print("[REQUEST] OpenAI-compatible endpoint called")
    return chat()

@app.route("/upload", methods=["POST"])
def upload_document():
    """Document upload endpoint with robust error handling"""
    try:
        print("[REQUEST] Document upload requested")
        
        # Validate file upload
        if 'file' not in request.files:
            print("[ERROR] No file in request")
            return jsonify({"success": False, "message": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            print("[ERROR] Empty filename")
            return jsonify({"success": False, "message": "No file selected"})
        
        # Validate file type
        if not file.filename.lower().endswith('.txt'):
            print(f"[ERROR] Invalid file type: {file.filename}")
            return jsonify({"success": False, "message": "Only .txt files are supported"})
        
        # Save file
        file_path = os.path.join('documents', file.filename)
        os.makedirs('documents', exist_ok=True)
        file.save(file_path)
        print(f"[UPLOAD] File saved: {file_path}")
        
        # Process document with RAG engine
        rag_engine = RAGEngine()
        success = rag_engine.add_document(file_path)
        
        if success:
            message = f"File '{file.filename}' uploaded and processed successfully"
            print(f"[UPLOAD] Success: {message}")
            return jsonify({"success": True, "message": message})
        else:
            message = f"File '{file.filename}' uploaded but processing failed"
            print(f"[UPLOAD] Processing failed: {message}")
            return jsonify({"success": False, "message": message})
            
    except Exception as e:
        error_message = f"Upload failed: {str(e)}"
        print(f"[ERROR] Upload error: {error_message}")
        return jsonify({"success": False, "message": error_message})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if RAG engine is initialized
        rag_engine = RAGEngine()
        embeddings_count = len(rag_engine.embeddings)
        
        return jsonify({
            "status": "healthy",
            "embeddings_loaded": embeddings_count,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

# Error handlers for Flask
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    print(f"[ERROR] 404 error: {request.url}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"[ERROR] 500 error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    try:
        print("[INIT] Starting Flask application...")
        
        # Initialize RAG engine at startup
        print("[INIT] Initializing RAG engine...")
        RAGEngine()
        print("[INIT] RAG engine ready")
        
        # Start Flask server
        print("[INIT] Starting Flask server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
        
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        raise