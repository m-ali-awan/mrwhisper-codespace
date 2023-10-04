# Imports
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate
import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sys
sys.path.append('/home/ubuntu/workspace/Creds/')
from openai_config import OPENAI_API_KEY

# creds and variables


# Custom image for the app icon and the assistant's avatar
company_logo = '/home/ubuntu/workspace/mrwhisper-codespace/ChatBot-Work/web_app/assets/SCR-20231002-owdl.png'

# Configure Streamlit page
st.set_page_config(
    page_title="MisterWhisper Chatbot",
    page_icon=company_logo
)

from help import *
with open('/home/ubuntu/workspace/mrwhisper-codespace/ChatBot-Work/web_app/authentication_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')
openai.api_key = OPENAI_API_KEY

         
if "messages" not in st.session_state:
    st.session_state.messages = []







if authentication_status:
    
    #st.write(f'Welcome *{name}*')
    st.title('MisterWhisper ChatBot')

    chain = load_chain()
    
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    prompt = st.chat_input("Ask AnyThing?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    #result = chain({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
    print(prompt)
    result = chain({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
    #result = chain("Give me code for how to load text documents")
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = result["answer"]
        message_placeholder.markdown(full_response + "|")
    message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    authenticator.logout('Logout', 'main')
    
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')