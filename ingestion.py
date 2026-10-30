import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Loading envs
load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")


# Creating client
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

# Delete collection if it already exists
if qdrant_client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    qdrant_client.delete_collection(COLLECTION_NAME)

# Creating collection
qdrant_client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# Load our knowledge
with open("my_facts.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents")

# Creating embedding
print("Loading embedding model")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

embeddings = embed_model.encode(documents)
print(f"Generated {len(embeddings)} embeddings")

# Creating qdrant points
points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i+1,
        vector=embedding.tolist(),
        payload={
            "text": documents[i]
        }
    )
    points.append(point)

# upload qdrant
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} document to qdrant!")
