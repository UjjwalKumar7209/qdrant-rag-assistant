import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Loading envs
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")


# Creating client of qdrant and groq
groq_client = Groq(api_key=groq_api_key)
groq_model = "openai/gpt-oss-120b"
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

