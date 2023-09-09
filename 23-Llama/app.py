from dotenv import load_dotenv
import streamlit as st
from utils.siteloader import (
    create_embeddings,
    pull_from_pinecone,
    get_similar_docs,
    get_answer
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
    st.set_page_config(page_title="ZulligPharma Large Language Model", page_icon=":robot_face:")
    st.header("Zuellig Automatic Ticket Classification Tool")
    st.session_state['Pinecone_API_Key']= st.text_input("What's your Pinecone API key?",type="password")
    
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:

        #creating embeddings instance
        embeddings=create_embeddings()

        #Function to pull index data from Pinecone
        index= pull_from_pinecone(st.session_state['Pinecone_API_Key'],constants.PINECONE_ENVIRONMENT,constants.PINECONE_INDEX,embeddings)
        
        #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        relavant_docs=get_similar_docs(index,user_input)

        #This will return the fine tuned response by LLM
        response=get_answer(relavant_docs,user_input)
        st.write(response)

        
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



