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

# Loading embed model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

def search_from_qdrant(query, top_k=3):
    query_vector = embed_model.encode(query).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results

def ask_llm(question, context):
    prompt = f"""
    Answer the question by using only the following given information below in one line:
    context:
    {context}

    Questions:
    {question}
    
    If the answer is not present in the context, say:
    "I don't know based on the provided information."
    """

    message = {
        "role": "user",
        "content": prompt
    }
    messages = [message]
    response = groq_client.chat.completions.create(model=groq_model,messages=messages)
    answer = response.choices[0].message.content
    return answer
    


# Questions from the user (RAG Pipeline)
while True:
    print("##################################################")
    user_input = input("Write your question (Enter exit to exit the chat.)\n")
    if user_input == "exit":
        break;
    results = search_from_qdrant(query=user_input, top_k=3)
    context = "\n".join(
        result.payload["text"]
        for result in results
    )
    answer = ask_llm(question=user_input, context=context)
    print(answer)
