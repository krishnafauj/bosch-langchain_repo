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
app = Flask(__name__)
CORS(app)
# Initialize Mistral API
MISTRAL_API_KEY = "Ra3QeOQoIpgH3fp1JIyMmfuj1D6FvX9K"  # Replace with your Mistral AI API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Define the template for the chatbot prompt
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

# Load the precomputed vector database
with open("vector_db.pkl", "rb") as f:
    vector_db = pickle.load(f)

# Initialize the embedding model
embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to call Mistral API
def call_mistral_api(messages):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-medium",  # Adjust model as necessary
        "messages": messages,
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Function to search the vector database
def search_vector_db(query: str, top_k: int = 3):
    # Embed the query
    query_embedding = embeddings.encode(query).tolist()

    # Compute similarities
    similarities = []
    for item in vector_db:
        similarity = cosine_similarity([query_embedding], [item["embedding"]])[0][0]
        similarities.append((similarity, item))

    # Sort by similarity and return top_k results
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in similarities[:top_k]]

# Function to generate a response using Mistral AI
def generate_response(query: str, text_extract: str) -> str:
    # Prepare the prompt for Mistral API
    system_message = {"role": "system", "content": prompt_template.format(text_extract=text_extract)}
    user_message = {"role": "user", "content": query}
    messages = [system_message, user_message]

    # Call Mistral AI API
    response = call_mistral_api(messages)
    return response["choices"][0]["message"]["content"]

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Search the vector database
    results = search_vector_db(question)
    text_extract = "\n".join([result["metadata"]["text"] for result in results])

    # Generate a response
    if text_extract.strip():  # Check if context is available
        answer = generate_response(question, text_extract)
    else:
        # Fallback response if no relevant context is found
        answer = "I don't have specific information on that topic, but generally, heavy-duty drilling requires powerful tools like hammer drills or rotary drills, along with appropriate drill bits and safety precautions."

    return jsonify({"answer": answer})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides a dynamic port

    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to all interfaces
