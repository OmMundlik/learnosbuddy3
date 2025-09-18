# Learnos Chatbot (RAG Based)

A simple RAG (Retrieval-Augmented Generation) chatbot using OpenAI, Pinecone, and Gradio.

## Setup

1. Install uv from https://docs.astral.sh/uv/
2. Copy `.env.example` to `.env` and add your API keys:
   - OpenAI API key
   - Pinecone API key and index name
3. Install dependencies:

```bash
uv sync
```

4. Index the tech news data:

```bash
uv run python index.py
```

5. Run the chatbot:

```bash
uv run python main.py
```

The chatbot will open in your browser at http://localhost:7860

## What it does

- **index.py**: Loads text files from `source_text/` folder, splits them into chunks, and stores in Pinecone
- **main.py**: RAG chatbot that searches Pinecone for relevant context and answers questions about recent tech news
- **source_text/**: Contains 5 recent tech news articles about AI developments


## To start using it type Hii!