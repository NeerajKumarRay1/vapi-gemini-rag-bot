# RAG Bot with Google Gemini

A Retrieval-Augmented Generation (RAG) chatbot built with Flask and Google's Gemini API. This application allows users to chat with an AI assistant that has access to a knowledge base of documents.

## Features

- Chat interface for interacting with the RAG-powered bot
- Document upload functionality to expand the knowledge base
- Vector-based retrieval for finding relevant information
- Powered by Google's Gemini API for high-quality responses

## Project Structure

```
vapi-gemini-rag-bot/
├── app.py               # Flask server
├── rag_engine.py        # RAG logic
├── vector_store/        # Stores your embeddings
├── documents/           # Knowledge base documents
│   ├── knowledge_base.txt
│   ├── calling_script.txt
├── templates/           # HTML templates
│   ├── index.html       # Chat interface
├── .env                 # Environment variables
├── requirements.txt     # Project dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository or download the source code

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables:
   - Copy the `.env` file and update it with your Gemini API key
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Using the RAG Bot

### Chatting with the Bot

1. Type your question in the chat input field
2. The bot will retrieve relevant information from the knowledge base and generate a response

### Adding Documents to the Knowledge Base

1. Use the upload form at the bottom of the page
2. Select a text file (.txt, .md, or .pdf)
3. Click "Upload Document"
4. The document will be processed and added to the knowledge base

## How It Works

1. **Document Processing**: When documents are uploaded, they are chunked and converted into vector embeddings using Gemini's embedding model.

2. **Query Processing**: When a user asks a question, the system:
   - Converts the question into an embedding
   - Finds the most similar document chunks using vector similarity
   - Retrieves the relevant text passages

3. **Response Generation**: The system then:
   - Combines the user's question with the retrieved passages
   - Sends this context-rich prompt to the Gemini model
   - Returns the generated response to the user

## Customization

- Modify the chunking parameters in `rag_engine.py` to adjust how documents are processed
- Update the prompt template in `process_query()` to change how the model generates responses
- Customize the UI by editing the HTML and CSS in `templates/index.html`

## License

This project is open source and available under the MIT License.

## Acknowledgements

- [Google Gemini API](https://ai.google.dev/)
- [Flask](https://flask.palletsprojects.com/)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)