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
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import *

from pathlib import Path
import sys
import os
sys.path.append('/home/ubuntu/workspace/Creds/')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n",".", '\n'],
    chunk_size=1500,
    chunk_overlap=100)


FILE_LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".ipynb": (NotebookLoader, {}),
    ".py": (PythonLoader, {}),
 
}

def load_docs_from_files(folder_path):
    all_files = os.listdir(folder_path)
    all_files = [i for i in all_files if not i.endswith('.ipynb_checkpoints')]
    print(all_files)
    loaded_documents = []
    for filename in all_files:
        #try:
            full_pth = Path(f'{folder_path}/{filename}')
            ext = os.path.splitext(full_pth.name)[-1][1:].lower()
            if ext in FILE_LOADER_MAPPING:
                loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                loader = loader_class(str(full_pth), **loader_args)
            else:
                loader = UnstructuredFileLoader(str(full_pth))
        #except:
        #    pass
        #try:
            loaded_documents.extend(loader.load())
            splitted_docs = text_splitter.split_documents(loaded_documents)
        #except:
        #    pass
    return splitted_docs



template = """
You are a customized Chatbot, and will be helping in different domains; like code helping, omniverse stuff, ShotGrid docs, api stuff etc. 
You will have memory, and previous messages will be provided to you in following format: \n
### Input  <Human Message>
### Response <Your Reply> 
and also make use of Knowledge base, which will be provided to you with some workflow.
-- If any code is to be written, DO WRITE IT INSIDE ``` <CODE> ``` . This is very important, as this way, the UI will be able to display code in well formatted 
way, by using some post-processing.

{context}

this is the running chat-history:
{chat_history}

### Input:  {question}

### Response:
"""
prompt = PromptTemplate(
    input_variables = ['context', 'question', 'chat_history'], template = template
)



llm = ChatOpenAI(model = 'gpt-3.5-turbo-16k',
                openai_api_key= OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def load_chain():
    memory = ConversationBufferWindowMemory(
        k = 4,
        memory_key= 'chat_history',
        human_prefix = '### Input',
        ai_prefix = '### Response',
        input_key = 'question',
        output_key = 'output_text',
        return_messages = True
    )
    chain = load_qa_chain(llm,chain_type='stuff',prompt = prompt, memory = memory, verbose =True)
    return chain



@st.cache_resource(show_spinner=False)
def load_vectorstore():
    load_vstore = FAISS.load_local('/home/ubuntu/workspace/Temp/04Oct-FAISS', embeddings = embeddings)

    return load_vstore

    

    


vector_store = load_vectorstore()
chain = load_chain()
@st.cache_data
def get_answer(user_input,_vector_store = vector_store, _chain = chain):

    
    docs = vector_store.similarity_search(user_input)
    answer = chain.run({'input_documents':docs,'question':user_input})



    return answer


    

    
    
    
    