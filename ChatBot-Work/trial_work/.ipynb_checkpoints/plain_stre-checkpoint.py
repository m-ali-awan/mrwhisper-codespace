import streamlit as st
import os
import shutil
from pathlib import Path
import openai
from plain_help import *
from langchain.document_loaders.chatgpt import ChatGPTLoader


dest_folder = Path('/home/ubuntu/workspace/Temp/UPLOADED-DIR')


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
text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n",".", '\n'],
    chunk_size=1500,
    chunk_overlap=100)

if st.button('Start new Chat'):
    st.session_state.messages = []
    st.session_state.chat_history = ''
    st.session_state.running_chat_summary = ''
    

prompt = st.chat_input('Ask Anything?')


if len(st.session_state.messages) > 1:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

#print(f':::::::::::::;{st.session_state.running_chat_summary}')

        

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
        # here with summary, only last 2 full conversations should go
        last_two_conversations = get_last_two_conversations(st.session_state.messages)
        chat_summary = running_summarize(st.session_state.running_chat_summary, last_two_conversations)
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


with st.sidebar:

    with st.form("Upload ChatGpt JSON export here", clear_on_submit=True):
        extension = 'json'
        uploaded_files = st.file_uploader('Upload your ChatGpt JSON export here', type = ['.json',], accept_multiple_files = True)
        button = st.form_submit_button('Submit!')
        if uploaded_files:
            if button:
                time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
                folder_name = f'{extension.upper()}-Documents-{time_now}'
                dir_path = dest_folder/folder_name
                dir_path.mkdir(exist_ok=True, parents=True)
                for uploaded_file in uploaded_files:
                    #bytes_data = uploaded_file.read()
                    with open(os.path.join(str(dir_path), uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())         
                    st.success("Saved File")
                docs = document_text_extraction('json',str(dir_path/uploaded_file.name))
                
                load_vstore = update_vectorstore_docs(load_vstore,docs)
    with st.form("Upload Google Docx here", clear_on_submit=True):
        uploaded_files = st.file_uploader('Upload Google Docx', type = ['.docx'], accept_multiple_files = False)
        button = st.form_submit_button('Submit!')
        if uploaded_files:
            if button:
                docs = document_text_extraction('docx',dest_folder, uploaded_files)
                
                load_vstore = update_vectorstore_docs(load_vstore,docs)
    with st.form("Upload PDF here", clear_on_submit=True):
        uploaded_files = st.file_uploader('Upload pdf', type = ['.pdf'], accept_multiple_files = False)
        button = st.form_submit_button('Submit!')
        if uploaded_files:
            if button:
                docs = document_text_extraction('pdf',dest_folder,uploaded_files)
                
                load_vstore = update_vectorstore_docs(load_vstore,docs)




    
    if len(st.session_state.messages)>2:
        st.title('Running Summary')
        st.markdown(st.session_state.running_chat_summary)
        if st.button('Add to VectorStore?'):
            document_to_add = [st.session_state.running_chat_summary]
            docs  = text_splitter.split_text(document_to_add)
            load_vstore = update_vectorstore(load_vstore,document_to_add)