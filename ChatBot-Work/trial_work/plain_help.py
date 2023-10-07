import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json
import os
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/workspace/Creds')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002',
                              openai_api_key= OPENAI_API_KEY)
load_vstore = FAISS.load_local('/home/ubuntu/workspace/Temp/04Oct-FAISS', embeddings = embeddings)


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
            your running summary(at that point) will be saved to a vectorstore. So the purpose is, for Example, user is chatting about a rcent python library code. and now over few iterations
            of conversations, you both were able to get the working code/config. Now the user wants to save this in vectorstore, so later anyone asks somethingspecifc/related to that, 
            because of your rich-content summary saved, Chatbot will be able to answer correctly maybe in first attempt, or atleast it will improve with time. So you will have Current
            -summary, new lines of conversations, and you have to generate new Summary. :: 
        
            '''},
            
            {'role':'user','content':user_message}
            
            
        ]

    
    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=message,
                            temperature=0.7
                                    )
    return response['choices'][0].message.content


def return_context_docs(query, vstore = load_vstore):
    docs = vstore.similarity_search(query)
    final_str = ''
    for one in docs:
        final_str += one.page_content
        final_str +='\n\n\n\n'

    return final_str



def main_chat_fn(query, vstore, chat_summary):

    context = return_context_docs(query,load_vstore)
    

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
                            model="gpt-3.5-turbo-16k",
                            messages=message,
                            temperature=0.7
                                    )
    return response['choices'][0].message.content


    