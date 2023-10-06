import streamlit as st
import os
import shutil
from pathlib import Path
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

import sys
sys.path.append('/home/ubuntu/workspace/Creds')
from openai_config import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

template = """ You are a chatbot, that has memory, with older memory being summarized dynamically. So, you have answer with best knowledge, and using the previous context:
Context:
---
{chat_history}
---
### Input : 
{question}

### Response:

"""
prompt = PromptTemplate(
    input_variables = ['question', 'chat_history'], template = template
)

llm = ChatOpenAI(model = 'gpt-3.5-turbo-16k',
                openai_api_key= OPENAI_API_KEY)

@st.cache_resource(show_spinner = True)
def return_chain():
    conversation_with_summary = ConversationChain(
        llm=llm,
        # We set a very low max_token_limit for the purposes of testing.
        memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=200),
        verbose=True,
    )
    return conversation_with_summary
