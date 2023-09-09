import streamlit as st
from utils.utils import *
from PIL import Image

# Define the Streamlit app
def main():
    st.title("Call Center Text Sentiment Analysis")
    st.image(Image.open("images/sentiment.jpg"), caption='Call Center Text Sentiment Analysis')




# Run the Streamlit app
if __name__ == "__main__":
    main()
