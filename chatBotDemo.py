import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#------------------------------------------------------------------------------
#page configuration
#------------------------------------------------------------------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("ask any question related to C++ Introduction:")
#------------------------------------------------------------------------------
#cache the data loading and processing
@st.cache_data
def load_vectorstore():
    #load the data
    loader=TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents=loader.load()
    #split the data
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    final_documents=text_splitter.split_documents(documents)      
    #embeding
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # create FAISS vector store        
    db=FAISS.from_documents(final_documents,embeddings)
    return db

#load vector DB (only once)
db=load_vectorstore()

#------------------------------------------------------------------------------
#user query and similarity search
query=st.text_input("Enter your question for C++: ")

docs=db.similarity_search(query)
st.subheader("retrieved contexts:")
for i, doc in enumerate(docs):
    st.markdown(f"**Result {i+1}:**")
    st.write(doc.page_content)

