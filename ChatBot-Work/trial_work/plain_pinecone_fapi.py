from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
from pathlib import Path
import openai
from plain_pinecone_help import *
from langchain.document_loaders.chatgpt import ChatGPTLoader
from langchain.vectorstores import Pinecone
import pinecone
from openai_config import OPENAI_API_KEY, MW_ENVIRONMENT, MW_PINECONE_API_KEY
from plain_pinecone_help import *
from pydantic import BaseModel
#

dest_folder = Path('/home/ubuntu/workspace/Temp/UPLOADED-DIR')

pinecone.init(
    api_key= MW_PINECONE_API_KEY,  # find at app.pinecone.io
    environment= MW_ENVIRONMENT,  # next to api key in console
)
pinecone_mw_index_name = 'minimal-index-dynamic-chatbot'

pinecone.init(
    api_key= MW_PINECONE_API_KEY,  # find at app.pinecone.io
    environment= MW_ENVIRONMENT,  # next to api key in console
)


#pinecone_index = Pinecone.get_pinecone_index(pinecone_mw_index_name)   This is raw pinecone index
pinecone_index = Pinecone.from_existing_index(pinecone_mw_index_name,embeddings)


app = FastAPI()

class AskModel(BaseModel):
    query: str
    chat_summary: str

class GetSummary(BaseModel):
    current_summary : str
    last_two_conversations : str

class Summary(BaseModel):

    summary : str

@app.post("/ask/")
def ask(request_body: AskModel):
    prompt = request_body.query
    chat_summary = request_body.chat_summary
    answer = main_chat_fn(prompt, pinecone_index, chat_summary)
    return {"response": answer}

@app.post("/upload/")
def upload_file(file: UploadFile = File(...), file_type: str = 'json'):
    # Handle file uploads based on file_type (json, docx, pdf)
    # ... your logic here ...
    return {"status": "File uploaded successfully"}

@app.post("/summary/")
def get_summary(request_body : GetSummary):
    # Return the running summary
    running_summarize(current_summary, last_two_conversations)
    return {"summary": running_chat_summary}

@app.post("/upsert/")
def upsert_summary(summary : Summary):
    # Upsert the running summary to Pinecone vectorstore
    # ... your logic here ...
    docs  = text_splitter.split_text(summary)
    load_vstore = update_pinecone_texts(pinecone_index,docs,pinecone_mw_index_name)
    return {"status": "Summary upserted successfully"}
