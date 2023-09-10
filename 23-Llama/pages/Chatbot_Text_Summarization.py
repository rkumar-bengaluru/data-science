import streamlit as st
from utils.utils import *
from PIL import Image
from langchain.schema import Document

# Define the Streamlit app
def main():
    
    
    st.title("Text Summarization...💁 ")

    st.write('https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML')
    st.subheader("I can help you on doc summarization")

    job_description = st.text_area("Please paste the 'Sample Document' here...",key="1")
    submit=st.button("Help me Summarize the doc")

    if submit:
        with st.spinner('Wait for it...'):

            # summarize the doc
            doc = Document(
                page_content=job_description,
                metadata={"name": "user_input","id":"user_input","type=":"text"},
            )
            summary = get_summary(doc)

            st.write("**Summary** : "+summary)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

        st.success("Hope I was able to save your time❤️")



'''
Refer to text files in audio directory, this were created from audio file.
'''

# Run the Streamlit app
if __name__ == "__main__":
    main()
