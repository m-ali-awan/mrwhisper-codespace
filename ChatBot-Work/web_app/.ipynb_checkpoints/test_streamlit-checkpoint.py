import streamlit as st


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.get('role')):
        st.write(message.get('content'))


prompt = st.chat_input('Ask Something')

if prompt:
    print(type(prompt))
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.write(prompt)