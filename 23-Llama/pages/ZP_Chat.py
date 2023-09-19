from utils.zpmodel import process_example
import streamlit as st

examples=[
            'How do I change language settings in ezrx portal?'
        ]
response = process_example(examples)
st.write(response)
