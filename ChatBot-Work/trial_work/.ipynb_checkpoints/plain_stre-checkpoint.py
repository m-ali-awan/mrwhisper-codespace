import streamlit as st
import os
import shutil
from pathlib import Path
import openai
from plain_help import *



# defining the session_states
if 'vectorstore_require_update' not in st.session_state:
    st.session_state.vectorstore_require_update = False

if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ''

if 'running_chat_summary' not in st.session_state:
    st.session_state.running_chat_summary = ''


st.title('MrWhisperTrial-ChatBot')

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= OPENAI_API_KEY)
load_vstore = FAISS.load_local('/home/ubuntu/workspace/Temp/04Oct-FAISS', embeddings = embeddings)

prompt = st.chat_input('Ask Anything?')


if len(st.session_state.messages) > 1:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

print(f':::::::::::::;{st.session_state.running_chat_summary}')
if len(st.session_state.messages)>2:
    with st.sidebar:
        st.title('Running Summary')
        st.markdown(st.session_state.running_chat_summary)

if prompt: 
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        

        
    if len(st.session_state.messages) > 1:
        chat_history = ''
        for message in st.session_state.messages:
            if message['role'] == 'user':
                chat_history += f"User :: {message['content']} \n"
            elif message['role'] == 'assistant':
                chat_history += f"AI Reponse :: {message['content']} \n"
        chat_summary = running_summarize(st.session_state.running_chat_summary, chat_history)
        st.session_state.running_chat_summary = chat_summary
    else:
        chat_summary = ''
    print(f"CHAT-SUMMARY::::::::::::\n")
    #print(chat_summary)
    answer = main_chat_fn(prompt,load_vstore,chat_summary)
    with open('/home/ubuntu/workspace/Temp/07-14Oct/testing_summaries.txt','a') as f:
        f.write(chat_summary)
        f.write('\n\n\n ----------------------------------- \n\n\n')
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message('assistant'):
        st.markdown(answer)