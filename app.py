from flask import Flask, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
import re
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
MISTRAL_API_KEY = "gbDR18JGRwoLDXAuFEpml7AufaKmjkqu"  # Replace with your Mistral AI API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers users' questions based on the context provided.

    Keep your answer short and to the point.
    
    The evidence is the context of the text extract with metadata. 
    
    Carefully focus on the metadata, especially 'filename' and 'page', whenever answering.
    
    Make sure to add filename and page number at the end of the sentence you are citing.
        
    Reply "Not applicable" if the text is irrelevant.
     
    The text content is:
    {text_extract}
"""

# Initialize embeddings for vector storage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize conversation history and answer storage
question_history = []  # Stores vector embeddings of questions
answer_history = []    # Stores corresponding answers

def parse_text(file_content: str, filename: str) -> Tuple[List[str], str]:
    text = file_content
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return [text], filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks  

def docs_to_index(docs):
    index = FAISS.from_documents(docs, embeddings)
    return index

def get_index_for_text(text_files, text_names):
    documents = []
    for text_file, text_name in zip(text_files, text_names):
        text, filename = parse_text(text_file, text_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents)
    return index

def load_text_file(file_path):
    with open(file_path, "r", encoding="latin-1") as file:
        content = file.read()
    return content

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

def is_similar_question(new_question: str, threshold: float = 0.8) -> Tuple[bool, str]:
    if not question_history:
        return False, ""
    new_question_embedding = embeddings.embed_query(new_question)
    similarities = cosine_similarity([new_question_embedding], question_history)[0]
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[max_similarity_index]
    if max_similarity > threshold:
        return True, answer_history[max_similarity_index]
    else:
        return False, ""
x
text_file_path = "./bosch_txt.txt"  # Replace with your text file path
text_content = load_text_file(text_file_path)
vectordb = get_index_for_text([text_content], ["text_file_name"])

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    is_similar, previous_answer = is_similar_question(question)
    if is_similar:
        return jsonify({"answer": previous_answer})
    search_results = vectordb.similarity_search(question, k=3)
    text_extract = "\n".join([result.page_content for result in search_results])
    system_message = {"role": "system", "content": prompt_template.format(text_extract=text_extract)}
    user_message = {"role": "user", "content": question}
    messages = [system_message, user_message]
    response = call_mistral_api(messages)
    answer = response["choices"][0]["message"]
    question_embedding = embeddings.embed_query(question)
    question_history.append(question_embedding)
    answer_history.append(answer)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
