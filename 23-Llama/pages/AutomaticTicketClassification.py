from dotenv import load_dotenv
import streamlit as st
from utils.siteloader import (
    create_embeddings,
    pull_from_pinecone,
    get_similar_docs,
    get_answer,
    predict
)
import constants


#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]


def main():
    load_dotenv()
    st.header("Zuellig Automatic Ticket Classification Tool")
    st.session_state['Pinecone_API_Key']= st.text_input("What's your Pinecone API key?",type="password")

    st.write('https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML')
    st.write('https://app.pinecone.io/')
    st.write('https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html')
    
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    st.write("Try --> What is the fare for transportation")
    user_input = st.text_input("🔍")

    if user_input:

        #creating embeddings instance
        embeddings=create_embeddings()

        #Function to pull index data from Pinecone
        print("pulling index...")
        index= pull_from_pinecone(st.session_state['Pinecone_API_Key'],constants.PINECONE_ENVIRONMENT,constants.PINECONE_INDEX,embeddings)
        print("pulling index done...")
        #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        print("getting similarity search...")
        relavant_docs=get_similar_docs(index,user_input)
        print("getting similarity search done...")
        print(relavant_docs)

        #This will return the fine tuned response by LLM
        #response=get_answer(relavant_docs,user_input)
        st.write(relavant_docs)
        
        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            #loading the ML model, so that we can use it to predit the class to which this compliant belongs to...
            department_value = predict(query_result)
            st.write("your ticket has been sumbitted to : "+department_value)

            #Appending the tickets to below list, so that we can view/use them later on...
            if department_value=="HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value=="IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)



if __name__ == '__main__':
    main()



