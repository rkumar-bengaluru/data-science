import streamlit as st
from utils.utils import *
from PIL import Image

# Define the Streamlit app
def main():
    st.title("Document Summarization")
    sunglasses = Image.open("images/summarization.png")
    st.image(sunglasses, caption='Sunrise by the mountains')




# Run the Streamlit app
if __name__ == "__main__":
    main()
