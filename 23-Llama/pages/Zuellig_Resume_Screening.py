import streamlit as st
from dotenv import load_dotenv
from utils.siteloader import create_embeddings, push_to_pinecone, get_resume_similar_docs, create_resume_docs
import uuid
from utils.utils import get_summary
import constants


def main():
    load_dotenv()
    st.title("HR - Resume Screening Assistance...💁 ")
    st.write('https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML')
    st.write('https://app.pinecone.io/')
    st.subheader("I can help you in resume screening process")

    st.session_state['Pinecone_API_Key']= st.text_input("What's your Pinecone API key?",type="password")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1")
    document_count = st.text_input("No.of 'RESUMES' to return",key="2")
    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_resume_docs(pdf,st.session_state['unique_id'])

            #Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            #Create embeddings instance
            embeddings=create_embeddings()

            #Push data to PINECONE
            push_to_pinecone(st.session_state['Pinecone_API_Key'],
                         constants.PINECONE_ENVIRONMENT,
                         constants.PINECONE_INDEX,embeddings,final_docs_list)

            #Fecth relavant documents from PINECONE
            relavant_docs=get_resume_similar_docs(
                job_description,
                document_count,
                st.session_state['Pinecone_API_Key'],
                constants.PINECONE_ENVIRONMENT,
                constants.PINECONE_INDEX,
                embeddings,
                st.session_state['unique_id'])

            st.write(relavant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Show me 👀'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    #st.write("***"+relavant_docs[item][0].page_content)
                    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time❤️")

    
'''
Qualifications

Bachelors of degree in computer science or related field
At least five year of experience working in software development 
Understanding of computer architecture, programming language and interface technology
'''

if __name__ == '__main__':
    main()



