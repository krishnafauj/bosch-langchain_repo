import requests
import json
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings  # Use HuggingFace embeddings as an alternative
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import faiss
import re
import numpy as np
from typing import List, Tuple
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
# Initialize Flask app
app = Flask(__name__)

# Mistral API configuration
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

# Function to parse text files
def parse_text(file_content: str, filename: str) -> Tuple[List[str], str]:
    text = file_content
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return [text], filename

# Function to convert text to documents
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

# Function to create an index from documents
def docs_to_index(docs):
    index = FAISS.from_documents(docs, embeddings)
    return index

# Function to get the index for text files
def get_index_for_text(text_files, text_names):
    documents = []
    for text_file, text_name in zip(text_files, text_names):
        text, filename = parse_text(text_file, text_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents)
    return index

# Load your text file-based database
def load_text_file(file_path):
    with open(file_path, "r", encoding="latin-1") as file:  # Specify Latin-1 encoding
        content = file.read()
    return content

# Function to call Mistral AI API
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

# Function to check if a question is similar to previous questions
def is_similar_question(new_question: str, threshold: float = 0.8) -> Tuple[bool, str]:
    if not question_history:
        return False, ""
    
    # Embed the new question
    new_question_embedding = embeddings.embed_query(new_question)
    
    # Compare with previous questions
    similarities = cosine_similarity([new_question_embedding], question_history)[0]
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[max_similarity_index]
    
    if max_similarity > threshold:
        return True, answer_history[max_similarity_index]
    else:
        return False, ""

# Load the text file and create the vector database
text_file_path = "/kaggle/input/bosch-txt-file-1/bosch_txt.txt"  # Replace with your text file path
text_content = load_text_file(text_file_path)
vectordb = get_index_for_text([text_content], ["text_file_name"])

# Flask endpoint to handle questions
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Check if the question is similar to previous questions
    is_similar, previous_answer = is_similar_question(question)
    if is_similar:
        return jsonify({"answer": previous_answer})
    
    # Search the vectordb for similar content to the user's question (RAG)
    search_results = vectordb.similarity_search(question, k=3)
    text_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the text extract
    system_message = {"role": "system", "content": prompt_template.format(text_extract=text_extract)}
    user_message = {"role": "user", "content": question}

    # Prepare the messages for the API call (CAG)
    messages = [system_message, user_message]

    # Call Mistral AI API and display the response
    response = call_mistral_api(messages)
    answer = response["choices"][0]["message"]["content"]

    # Clean the answer to remove unwanted references or formatting
    clean_answer = re.sub(r'\([^)]*\)', '', answer)  # Remove parentheses and their content
    clean_answer = re.sub(r'\{[^}]*\}', '', clean_answer)  # Remove curly braces and their content
    clean_answer = clean_answer.strip()  # Remove any leading/trailing whitespace

    # Store the question and answer in history
    question_embedding = embeddings.embed_query(question)
    question_history.append(question_embedding)
    answer_history.append(clean_answer)

    return jsonify({"answer": clean_answer})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
