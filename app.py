from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import requests
from rag_engine import query_context

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"]) 
def chat(): 
    try: 
        data = request.get_json() 
        print("💡 Received from Vapi:", data) 

        # Custom LLM mode: get user input from last message 
        if "messages" in data and len(data["messages"]) > 0: 
            user_input = data["messages"][-1]["content"] 
        else: 
            user_input = data.get("transcript", "") 
        print("📝 User input:", user_input) 

        context = query_context(user_input) 
        print("📚 Context:", context) 

        prompt = f"""You are Veena, an insurance advisor. Use the context below. 
Context: 
{context} 

User: {user_input} 
Veena:"""

        payload = { 
            "contents": [ 
                {"role": "user", "parts": [{"text": prompt}]} 
            ] 
        } 

        response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=payload) 
        print("🌐 Gemini raw response:", response.text) 

        data = response.json() 
        if "candidates" in data: 
            reply = data["candidates"][0]["content"]["parts"][0]["text"] 
        else: 
            reply = "I'm sorry, I couldn't get an answer from Gemini." 

        print("✅ Reply to Vapi:", reply) 
        return jsonify({"text": reply}) 

    except Exception as e: 
        print("❌ Exception:", str(e)) 
        # Always return some text so Vapi doesn't fail 
        return jsonify({"text": f"Error: {str(e)}"})

@app.route("/chat/completions", methods=["POST"]) 
def chat_completions(): 
    return chat()

@app.route("/upload", methods=["POST"])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
    
    if file and file.filename.endswith('.txt'):
        file_path = os.path.join('documents', file.filename)
        file.save(file_path)
        
        # Process the new document
        from rag_engine import RAGEngine
        engine = RAGEngine()
        engine.add_document(file_path)
        
        return jsonify({"success": True, "message": f"File {file.filename} uploaded and processed successfully"})
    else:
        return jsonify({"success": False, "message": "Only .txt files are supported"})

if __name__ == "__main__":
    from rag_engine import create_vectorstore
    create_vectorstore()
    app.run(host="0.0.0.0", port=5000)