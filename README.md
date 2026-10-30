# Qdrant RAG Assistant

A small retrieval-augmented generation (RAG) assistant built with Python, Qdrant, Sentence Transformers, and Groq.

The project loads facts from `my_facts.txt`, converts each line into a vector using the `all-MiniLM-L6-v2` embedding model, and stores the vectors in a Qdrant collection. When you ask a question, the assistant retrieves the three most relevant facts and sends them to a Groq language model to generate an answer.

## Requirements

- Python 3.11 or newer
- A Qdrant Cloud account or accessible Qdrant instance
- A Groq API key

## Setup

1. Create and activate a virtual environment:

	```bash
	python -m venv .venv
	source .venv/bin/activate
	```

2. Install the project dependencies:

	```bash
	pip install -e .
	```

3. Create a `.env` file in the project root:

	```env
	GROQ_API_KEY=your_groq_api_key
	QDRANT_URL=https://your-qdrant-instance-url
	QDRANT_API_KEY=your_qdrant_api_key
	```

## Load the knowledge base

Run the ingestion script before starting the assistant:

```bash
python ingestion.py
```

This creates the `knowledge` collection and uploads the non-empty lines from `my_facts.txt`. Running the script deletes and recreates that collection, so any existing data in `knowledge` will be replaced.

## Start the assistant

```bash
python main.py
```

Enter a question at the prompt. The assistant uses only the retrieved facts when generating its response. Type `exit` to end the chat.

## Project structure

```text
.
├── ingestion.py       Load facts and create Qdrant vectors
├── main.py             Run the interactive RAG assistant
├── my_facts.txt       Source knowledge base
├── pyproject.toml      Project metadata and dependencies
└── src/
	 └── qdrant_rag_assistant/
		  └── __init__.py
```

## Customizing the knowledge base

Replace or extend the contents of `my_facts.txt`, keeping one fact per line. Run `python ingestion.py` again to rebuild the Qdrant collection with the updated content.
