from typing import Iterator

import torch

from utils.model import get_input_token_length, run
import streamlit as st
from streamlit_chat import message

DESCRIPTION = """
# Llama-2 7B Chat
This Space demonstrates model [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta, a Llama 2 model with 7B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).
🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).
🔨 Looking for an even more powerful model? Check out the [13B version](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat) or the large [70B model demo](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI).
"""
LICENSE = """
<p/>
---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        st.write(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')

def get_llm_response(user_input):

    if st.session_state['conversation'] is None:
        response = generate(user_input, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
        print(response)
        for x in response:
            print('', x)
        
        #llm = get_transformer()
        # st.session_state['conversation'] = langchain.chains.ConversationChain(
        #     llm=llm,
        #     verbose=True,
        #     memory= ConversationBufferMemory()
        # )
    
    #response = st.session_state['conversation'].predict(input=user_input)
    #print(st.session_state['conversation'].memory.buffer)
    return response

st.markdown(DESCRIPTION)

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