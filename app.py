from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import json
import aiofiles
from fastapi.encoders import jsonable_encoder
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/docs/'

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
index = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = index.as_retriever(search_type="similarity", search_kwargs={'k': 3})

llm = OpenAI(model_kwargs={"max_tokens": 500}, temperature=0.4)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["POST"])
def chat():
    msg = request.form['msg']
    response = rag_chain.invoke({'input': msg})
    return jsonify({"answer": response["answer"]})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
   
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
  
   
    print(f"Checking if file exists: {pdf_path}")
    if os.path.exists(pdf_path):
        print(f"File already exists: {pdf_path}")
        return jsonify({"msg": "File already exists, skipping upload", "pdf_filename": filename})

    
    file.save(pdf_path)
    print(f"File saved: {pdf_path}")


    try:
        index_name = "medicalbot"
        index = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        extracted_data=load_pdf_file('static/docs/')
        uploaded_chunks = text_split(extracted_data) 

        existing_vectors = index.similarity_search(filename, k=1)
        if existing_vectors:
            return jsonify({"msg": "File already exists in Pinecone, skipping upload", "pdf_filename": filename})
     
        index = PineconeVectorStore.from_documents(
            uploaded_chunks,
            embeddings,
            index_name=index_name
        )

        return jsonify({"msg": "success", "pdf_filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
