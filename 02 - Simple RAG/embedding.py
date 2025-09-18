"""
EMBEDDING.PY - Google Gemini Embedding Service

What it does:
- Provides a simple function to convert text into vector embeddings using Google Gemini
- Acts as a reusable module for both indexing and searching operations

How it works:
1. Initializes Google Gemini client with API key from environment variables
2. Takes any text string as input
3. Uses Gemini's embedding model to convert text into numerical vectors
4. Returns the embedding vector that can be stored in Pinecone or used for similarity search

Why we use this:
- Google Gemini embeddings are free to use
- Embeddings allow us to find semantically similar text chunks
- Centralized in one file so both index.py and main.py can use the same function
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    """Get embedding from Google Gemini"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values