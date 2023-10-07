import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json
import os
from pathlib import Path
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import *
from datetime import datetime


sys.path.append('/home/ubuntu/workspace/Creds')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= OPENAI_API_KEY)
load_vstore = FAISS.load_local('/home/ubuntu/workspace/Temp/07-14Oct/minimal_knowledgebase', embeddings = embeddings)


def running_summarize(chat_history, new_query):


    user_message =    f""" Current Summary:
    {chat_history}
    
    New lines of conversation :
    {new_query}
    
    New_Summary:

    """
    message = [
            {'role':'system','content':'''
            You are acting as dynamic memory of a chatbot(Chatbot: A customized one, for helping with latest python libraries, tools like omniverse, usd files etc). You have to be
            mindful, that in dynamic summary generation, you have to return the corrected code/ any configuration related stuff etc. As, there will be a button in USEr Interface, with which 
            your running summary(at that point) will be saved to a vectorstore. So the purpose is, for Example, user is chatting about a recent python library code. and now over few iterations
            of conversations, you both were able to get the working code/config. Now the user wants to save this in vectorstore, so later anyone asks somethingspecifc/related to that, 
            because of your SHORT-EFFECTIVE summary saved.  So you will have Current
            -summary, new lines of conversations, and you have to generate new Summary. 
            ******** DO INCLUDE IN CORRECTED/REQUIRED CODE OR CONFIG - AND BE LESS VERBOSE, FOCUS ON IMPORTANT THINGS ONLY *********:: 
        
            '''},
            
            {'role':'user','content':user_message}
            
            
        ]

    
    response = openai.ChatCompletion.create(
                            #model="gpt-3.5-turbo-16k",
                            model="gpt-3.5-turbo",
                            messages=message,
                            temperature=0.7
                                    )
    return response['choices'][0].message.content


def return_context_docs(query, vstore = load_vstore):
    '''
    will return full text, and also top doc text, for decision to use it or not

    '''
    docs = vstore.similarity_search(query)
    final_str = ''
    for one in docs:
        final_str += one.page_content
        final_str +='\n\n\n\n'

    return final_str, docs[0].page_content
    

def get_response_type(s):
    return s.split(":")[0].strip()

def main_chat_fn(query, vstore, chat_summary):

    context,decision_context = return_context_docs(query,load_vstore)
    to_use_context = decision_to_use_context(decision_context,query)
    to_use_context_decision = get_response_type(to_use_context)
    print(to_use_context)
    if to_use_context_decision == 'Yes':
        context = context
        print('USED:::')
    elif to_use_context_decision =='No':
        context = '''We don't have any relevant Information, so use your best knowledge, and if you don't know, say : i don't know. Kindly provide me some context, or internet access'''
        print(context)

    user_message = f"""
    Context of knowledgebase:
    -----
    {context}
    -----
    Running summary of chat:
    -----
    {chat_summary}
    ----
    Query : {query}

    AI Response:

    """
    message = [
        {'role':'system','content':'''You are a customized Chatbot, and will be helping in different domains; like code helping, omniverse stuff, ShotGrid docs, api stuff etc. 
You will have memory, and previous messages will be provided to you. Also, custom knowlwdgebase, similar chunks of info, will be provided as context.'''},
        {'role':'user','content':user_message}
        
        
    ]

    
    response = openai.ChatCompletion.create(
                            #model="gpt-3.5-turbo-16k",
                            model = 'gpt-4',
                            messages=message,
                            temperature=0.7
                                    )
    return response['choices'][0].message.content


def decision_to_use_context(docs,query):

    messages = [
        {'role':'system','content':"""
        We have large store of knowledgebase texts, and we have one api/fn that based on QUERY returns relevant chunks of texts. Now, it is error-prone, 
        like even if relevant info is not present in knowledgebase, it does return something.
        
        **Be mindful in cases of code, It should be returning docs of relevant library, 
        and methods, not of any random one. !! IF ERROR IS IN FIFTYONE DATASETS METHOD/OBJECT LIBRARY RELATED, CONTEXTS SHOULD NOT BE SOME CODE-RELATED TO ERROR/SOLVING ABOUT LANGCHAIN DATASETS/ETC,SHOTGRID ETC
        You have to make the decision , Whether than information is good to use or not.**
        
        You have to only Answer in "Yes:Small Reason" and "No: Small Reason". There is no option of Not knowing, or any explanation stuff.
         \n\n
                                """},
        {'role':'user','content':"""
        Query is : "got this error
        AttributeError Traceback (most recent call last) Cell In[8], line 1 ----> 1 classes = dst.classifications.classes 2 print(classes)
        
        File ~/.pyenv/versions/env-yolov8/lib/python3.10/site-packages/fiftyone/core/dataset.py:357, in Dataset.getattribute(self, name) 354 if getattr(self, "_deleted", False): 355 raise ValueError("Dataset '%s' is deleted" % self.name) --> 357 return super().getattribute(name)
        
        AttributeError: 'Dataset' object has no attribute 'classifications'", \n
        Context docs are : "# we drop sparse_values as they are not needed for this example\n\ndataset.documents.drop([\n\n\'metadata\'], axis\n\n1, inplace\n\nTrue)\n\ndataset.documents.rename(columns\n\n={\n\n\'blob\':\n\n\'metadata\'}, inplace\n\nTrue)\n\ndataset.head()\n\n0 \n       417ede5d-39be-498f-b518-f47ed4e53b90 \n       [0.005949743557721376, 0.01983247883617878, -0... \n       {\'chunk\': 0, \'text\': \'.rst\n.pdf\nWelcome to Lan... \n     \n       1 \n       110f550d-110b-4378-b95e-141397fa21bc \n       [0.009401749819517136, 0.02443608082830906, 0..\n\n\n\nGPT4 with\nRetrieval Augmentation over LangChain Docs\n\nIn this notebook we\'ll work through an example of using GPT-4 with\nretrieval augmentation to answer questions about the LangChain Python\nlibrary.\n\nTo begin we must install the prerequisite libraries:\n\n!pip install -qU \\\n\nopenai==0.27.7 \\\n\n"pinecone-client[grpc]"==2.2.1 \\\n\npinecone-datasets==\'0.5.0rc11\'\n\n🚨 Note: the above pip install is formatted for\nJupyter notebooks. If running elsewhere you may need to drop the\n!n\n\n\n"
        """},
        {'role':'assistant','content': "No: As query is for Fiftyone Related, and context is about Langchain, GPT4,etc"},


        {'role':'user','content':f"""
        Query is : "{query}" \n
        Context docs are : "{docs}"
        """}
    ]
    response = openai.ChatCompletion.create(
                            #model="gpt-3.5-turbo-16k",
                            model = 'gpt-4',
                            messages=messages,
                            temperature=0
                                    )
    return response['choices'][0].message.content


def update_vectorstore(load_vstore,document_to_add, path):
    load_vstore.add_texts(document_to_add)
    load_vstore.save_local(path)
    print('done')
    return load_vstore


def update_vectorstore_docs(load_vstore,document_to_add, path):
    load_vstore.add_documents(document_to_add)
    load_vstore.save_local(path)
    return load_vstore

text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n",".", '\n'],
    chunk_size=1500,
    chunk_overlap=100)

def update_pinecone_texts(pinecone_index, texts_to_add):
    
    index_name = pinecone_index.configuration.server_variables['index_name']
    updated_index = Pinecone.from_texts(texts_to_add,embeddings, index_name = index_name)
    return updated_index
def update_pinecone_docs(pinecone_index, docs_to_add):

    index_name = pinecone_index.configuration.server_variables['index_name']
    updated_index = Pinecone.from_documents(docs_to_add,embeddings, index_name = index_name)
    return updated_index
    
def document_text_extraction(extension,file_path):

    
    if extension == "json":
        loader = ChatGPTLoader(log_file=str(file_path), num_logs=0)
        docs = loader.load()
    elif extension == "txt":
        loader = TextLoader(str(file_path))
        docs = loader.load()
    elif extension =='pdf':
        loader = PyPDFLoader(str(file_path))
        docs = loader.load_and_split()
    elif extension == 'docx':
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
    
    docs = text_splitter.split_documents(docs)
    print(len(docs))


    return docs


def get_last_two_conversations(messages):
    # Ensure there are at least 4 messages
    if len(messages) < 4:
        # Return whatever messages are available
        last_messages = messages
    else:
        # Get the last 4 messages (2 conversations)
        last_messages = messages[-4:]
    
    result = ""
    for message in last_messages:
        if message["role"] == "user":
            result += f"User : {message['content']}\n"
        elif message["role"] == "assistant":
            result += f"AI ASSISTANT : {message['content']}\n\n"
    
    return result



