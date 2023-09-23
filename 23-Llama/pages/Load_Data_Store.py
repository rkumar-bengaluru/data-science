import streamlit as st
from dotenv import load_dotenv
from utils.utils import (get_pdf_text)

from utils.siteloader import split_data_pdf, create_embeddings, push_to_pinecone
import constants


def main():
    load_dotenv()

    st.title("Please upload your files...📁 ")

    st.session_state['Pinecone_tickets_key']= st.text_input("What's your Pinecone Tickets API key?",type="password")

    # Upload the pdf file
    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    # Extract the whole text from the uploaded pdf file
    if pdf is not None:
        with st.spinner('Wait for it...'):
            text=get_pdf_text(pdf)
            st.write("👉Reading PDF done")

            # Create chunks
            docs_chunks=split_data_pdf(text)
            #st.write(docs_chunks)
            st.write("👉Splitting data into chunks done")

            # Create the embeddings
            embeddings=create_embeddings()
            st.write("👉Creating embeddings instance done")

            # Build the vector store (Push the PDF data embeddings)
            push_to_pinecone(st.session_state['Pinecone_tickets_key'],
                         constants.PINECONE_ENVIRONMENT,
                         constants.PINECONE_TICKET_INDEX,embeddings,docs_chunks)

        st.success("Successfully pushed the embeddings to Pinecone")


if __name__ == '__main__':
    main()
