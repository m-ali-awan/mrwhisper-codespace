import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pathlib import Path
import sys
sys.path.append('/home/ubuntu/workspace/Creds/')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY



template = """
You are a customized Chatbot, and will be helping in different domains; like code helping, omniverse stuff, ShotGrid docs, api stuff etc. 
It is very important that you do have memory, and can go along with the User, and also make use of Knowledge base, which will be provided to you with some workflow.
-- If any code is to be written, DO WRITE IT INSIDE ``` <CODE> ``` . This is very important, as this way, the UI will be able to display code in well formatted 
way, by using some post-processing.

{context}

this is the running chat-history:
{chat_history}

Query:  {question}

Response:
"""
prompt = PromptTemplate(
    input_variables = ['context', 'question', 'chat_history'], template = template
)

memory = ConversationBufferWindowMemory(
    k = 4,
    memory_key = 'chat_history',
    return_messages = False
)

llm = ChatOpenAI(model = 'gpt-3.5-turbo-16k',
                openai_api_key= OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n",".", '\n'],
    chunk_size=1500,
    chunk_overlap=100)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    load_vstore = FAISS.load_local('/home/ubuntu/workspace/Temp/04Oct-FAISS', embeddings = embeddings)

    return load_vstore

    

    


vector_store = load_vectorstore()
@st.cache_data
def load_chain(vector_store = vector_store, llm = llm, memory = memory):

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                              retriever=retriever, 
                                              memory=memory, 
                                              #get_chat_history=lambda h : h,
                                              combine_docs_chain_kwargs = {'prompt':prompt},
                                              verbose=True)

    return chain


    

    
    
    
    