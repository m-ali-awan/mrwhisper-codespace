import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]



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
                openai_api_key= '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')

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= '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')

@st.cache_data
def load_chain():
    