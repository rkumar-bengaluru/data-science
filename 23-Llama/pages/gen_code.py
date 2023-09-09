import streamlit as st

import langchain

#from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers
import pandas as pd
import numpy as np
from utils.utils import get_transformer, get_prompt_template

def getLLMResponse(instruction_txt, input_txt):
    llm = get_transformer()
    template = """
    Below is an intruction that describes a task, paired with inout that provides further context. Write
    a response that appropriately completes the request.

    ### Instruction:    {instruction}
    ### Input:  {input}
    ### Response:  
    """

    prompt = get_prompt_template(template)
    response = llm(prompt.format(instruction=instruction_txt, input=input_txt))
    print(response)
    return response

st.set_page_config(page_title="Generate Code",
                   page_icon=",",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.header("Generate Code - Alpaca Dataset")
st.write('https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json')
df = pd.read_json('data/alpaca_master_data_code_alpaca_20k.json')
df = df.head()
st.table(df)
instruction_text = st.text_area("Enter Instruction:", height=100,key='instruction')
input_txt = st.text_area("Enter Input:", height=100,key='input_txt')

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(instruction_text, input_txt))