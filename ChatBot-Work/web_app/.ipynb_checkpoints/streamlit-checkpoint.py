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

# creds and variables
with open('authentication_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')
openai.api_key = st.secrets["OPENAI_API_KEY"]






if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    st.title('Some content')
    code_block = "python\nimport DaVinciResolveScript as bmd\n\n# Connect to the DaVinci Resolve project manager\nresolve = bmd.scriptapp('Resolve')\npm = resolve.GetProjectManager()\n\n# Create a new project\nproject = pm.CreateProject('My Project', '/path/to/project')\n\n# Set project settings\nsettings = project.GetSetting()\nsettings['timelineResolutionWidth'] = 1920\nsettings['timelineResolutionHeight'] = 1080\n\n# Save the project\nproject.Save()\n\n# Close the project manager\npm.CloseProjectManager()\n"
    st.code(code_block)
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')