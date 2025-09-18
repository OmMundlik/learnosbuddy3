"""
INDEX.PY - Document Indexing Script

What it does:
- Loads all text files from the source_text folder
- Converts each file into embeddings using Google Gemini
- Stores the embeddings in Pinecone vector database for fast similarity search

How it works:
1. Finds all .txt files in the source_text directory
2. Reads each file content
3. For each file:
   - Generates an embedding using Gemini
   - Creates a unique ID based on filename
   - Stores in Pinecone with the original text as metadata
4. Prints progress as it processes each file

Since our source files are small, we process each file as a single unit without chunking.
Run this once to index all your documents before using the chatbot.
"""

import os
import glob
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding import get_embedding

load_dotenv("../.env")

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def load_and_index_files():
    """Load all text files and index them in Pinecone"""
    txt_files = glob.glob("source_text/*.txt")
    
    for file_path in txt_files:
        print(f"Processing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        file_id = os.path.basename(file_path).replace('.txt', '')
        embedding = get_embedding(content)
        
        index.upsert(vectors=[{
            "id": file_id,
            "values": embedding,
            "metadata": {
                "text": content,
                "filename": os.path.basename(file_path)
            }
        }])
        
        print(f"Indexed {file_path} as single document")

if __name__ == "__main__":
    load_and_index_files()
    print("All files indexed successfully!")