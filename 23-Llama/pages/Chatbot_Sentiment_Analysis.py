import streamlit as st
from utils.utils import *
from PIL import Image
from transformers import pipeline

# Define the Streamlit app
def main():
    st.title("Call Center Text Sentiment Analysis")
    st.image(Image.open("images/sentiment.jpg"), caption='Call Center Text Sentiment Analysis')

    st.title("Text Sentiment Analysis...💁 ")

    st.write('This model ("SiEBERT", prefix for "Sentiment in English") is a fine-tuned checkpoint of RoBERTa-large (Liu et al. 2019). It enables reliable binary sentiment analysis for various types of English-language text. ')
    
    st.write('https://huggingface.co/siebert/sentiment-roberta-large-english')
    
    user_input = st.text_area("Please paste the 'Sample Text for Sentiment Analysis' here...",key="1")
    submit=st.button("Please do the analysis")

    if submit:
        with st.spinner('Wait for it...'):

            # summarize the doc
            sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
            st.write(sentiment_analysis(user_input))

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

        st.success("Hope I was able to save your time❤️")


'''
c1
Sharat contacted ABC Bangs Credit Card customer care to make an online payment towards his credit card bill. He made the 
payment successfully and confirmed that it was processed correctly.

c2
The text is a series of requests and instructions regarding an investigation process. It asks the reader to "feel free" in 
various ways throughout the process, such as feeling free to investigate, freeze the investigation, or don't notice it. The 
text also encourages the reader to let you will be sure and feel free to investigate.

c3
the mobile app can be really glitchy and is definately not user friendly

'''



# Run the Streamlit app
if __name__ == "__main__":
    main()
