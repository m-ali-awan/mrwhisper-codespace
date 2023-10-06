import streamlit as st
import os
import shutil
from pathlib import Path
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import sys

from trial_help import return_chain
# Creds
sys.path.append('/home/ubuntu/workspace/Creds')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

#
# defining the session_states
if 'vectorstore_require_update' not in st.session_state:
    st.session_state.vectorstore_require_update = False

if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

if 'messages' not in st.session_state:
    st.session_state.messages = []


st.title('MrWhisperTrial-ChatBot')

chain = return_chain()
prompt = st.chat_input('Ask Anything?')

if prompt: 
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown('prompt')

    answer = chain({'input':prompt})

    with st.chat_message('assistant'):
        st.markdown(answer)