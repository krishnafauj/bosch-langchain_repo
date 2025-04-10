from flask import Flask, request, jsonify
import json
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import requests
from flask_cors import CORS 
import os

app = Flask(__name__)
CORS(app)

# Initialize Mistral API
MISTRAL_API_KEY = "Z51xg0MS4qD9Q6NPGZjQrq9pbDeCsQ8E"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Prompt template
prompt_template = """
    You are a helpful Assistant who answers users' questions based on the context provided.
    In Any Case You will not give any other name other than Bosch.
    You are a chat bot for bosch company so reply like a dependent bot of bosch not just a random chat bot.
    If you don't find any relevant company, just say about Bosch Ignore all other things like a chat bot. 
    Keep your answer short and to the point.
    Don't Write the sentence based on the context provided.
    Stick with Bosch brand only.
    If the context is insufficient, provide a general answer based on your knowledge.
    The text content is:
    {text_extract}
"""

# Lazy-loaded vector DB
vector_db = None

# Initialize the embedding model (only once)
embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to lazy load vector DB
def load_vector_db_once():
    global vector_db
    if vector_db is None:
        print("📦 Loading vector database into memory...")
        with open("new_dataset.pkl", "rb") as f:
            vector_db = pickle.load(f)
    return vector_db

# Mistral API call
def call_mistral_api(messages):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-medium",
        "messages": messages,
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Vector DB search
def search_vector_db(query: str, top_k: int = 3):
    current_db = load_vector_db_once()
    query_embedding = embeddings.encode(query).tolist()
    similarities = []

    for item in current_db:
        similarity = cosine_similarity([query_embedding], [item["embedding"]])[0][0]
        similarities.append((similarity, item))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in similarities[:top_k]]

# Mistral response
def generate_response(query: str, text_extract: str) -> str:
    system_message = {"role": "system", "content": prompt_template.format(text_extract=text_extract)}
    user_message = {"role": "user", "content": query}
    messages = [system_message, user_message]
    response = call_mistral_api(messages)
    return response["choices"][0]["message"]["content"]

# Endpoint
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    results = search_vector_db(question)
    text_extract = "\n".join([result["metadata"]["text"] for result in results])

    if text_extract.strip():
        answer = generate_response(question, text_extract)
    else:
        answer = "I don't have specific information on that topic, but generally, heavy-duty drilling requires powerful tools like hammer drills or rotary drills, along with appropriate drill bits and safety precautions."

    return jsonify({"answer": answer})

# Preload DB at startup to avoid cold start lag
if __name__ == '__main__':
    load_vector_db_once()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
