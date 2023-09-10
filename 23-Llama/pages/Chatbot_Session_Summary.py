from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory)
import langchain
import streamlit as st
from streamlit_chat import message
import os
from langchain.llms import CTransformers
from PIL import Image
from utils.utils import get_transformer, get_prompt_template

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


st.markdown("<h1 style='text-align: center;'> How can I assist you?</h1>", unsafe_allow_html=True)
st.sidebar.title("😎")

sunglasses = Image.open("images/chatsummary.png")
st.write('https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML')
st.image(sunglasses, caption='ZuelligPharma Chatbot')

summarize_button = st.sidebar.button("Summarize the conversation", key="chatbot_summarize")

if summarize_button:
    summarise_placeholder = st.sidebar.write("Nice chatting with you my friend ❤️:\n\n"+st.session_state['conversation'].memory.buffer)




def get_llm_response(user_input):

    if st.session_state['conversation'] is None:
        llm = get_transformer()
        st.session_state['conversation'] = langchain.chains.ConversationChain(
            llm=llm,
            verbose=True,
            memory= ConversationBufferMemory()
        )
    
    response = st.session_state['conversation'].predict(input=user_input)
    print(st.session_state['conversation'].memory.buffer)
    return response


response_container = st.container()
request_container = st.container()

with request_container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("Type your question here:", key="chatbot_input", height=100)
        submit_button = st.form_submit_button(label="Send")

        if submit_button:
            st.session_state['messages'].append(user_input)
            response_from_bot = get_llm_response(user_input)
            st.session_state['messages'].append(response_from_bot)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')
                #st.write(response_from_bot)


# conversation("Good Morning All...")
# conversation("My name is rupak")
# conversation("I live in bangalore india")
# print(conversation.memory.buffer)
# conversation.predict(input="What is my name")