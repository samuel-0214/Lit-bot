# importing all the important libraries
import streamlit as st
from dotenv import load_dotenv,find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import Client, completions, OpenAI
import os 


st.set_page_config(page_title="Multiple-PDF-Bot", page_icon=":books:")

st.header("Chaduvukondi firstuu!!!:books:")

#reading the document
def get_pdf_text(pdf_docs):
    text = "" # initializing a variable to store the text that is read
    #iterating through the document and storing the whole document in pdf variable 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() # extracting the pages from the pdf that is read
    return text

#converting the text read into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #splitting the text 
    chunks = text_splitter.split_text(text)
    return chunks

#chunks of data into vectorDB
def get_vectorstore(text_chunks, api_key1):
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    #for loading the pdfs that are processed 
    load_dotenv(find_dotenv())

def get_conversational_chain(context, user_question):
    client = Client(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_template = "Please answer the question with reference to the document provided."

    # Incorporate the document context and the user question into the prompt
    prompt = f"Document: {context}\nQuestion: {user_question}\n{prompt_template}"

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use an appropriate model
        prompt=prompt,
        temperature=0.5,  # Adjust temperature as needed
        max_tokens=550  # Adjust max_tokens as needed
    )
    return response.choices[0].text

def user_input(user_question, document_text):
    # This part remains unchanged
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.environ.get("OPENAI_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Get the response from the OpenAI API
    response_text = get_conversational_chain(document_text, user_question)
    
    # Now, instead of trying to call `response_text` as a function,
    # you should process it according to your application's needs.
    # For example, you might log it, display it, or use it to inform further processing.
    st.write("Reply: ", response_text) 


api_key = os.environ.get("OPENAI_API_KEY")


def main():
    #for creating the website

    user_question = st.text_input("Ask a Question related to the PDF Files that you uploaded", key="user_question")
    document_text=""

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, document_text)
    

    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader("Upload your PDFs",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text 
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                #create vectorstore
                get_vectorstore(text_chunks, api_key)

                st.success("Done")

                #creating conversation chain
                #conversation = get_conversation_chain(vectorstores)
                

if __name__ == '__main__':
    main() 