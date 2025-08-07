import os
from re import match

import PyPDF2
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import tempfile
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY="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"
#Upload pdf

HUGGING_FACE_TOKEN = "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"






if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def extract_text_from_pdf(pdf_file):
    text = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text +=page.extract_text()

    os.unlink(tmp_file_path)
    return text


def create_vector_store(text):
    """Create Vector embeddings and store them in FAISS"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200
                                                   )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def setup_qa_chain(vector_store):
    """Setup Question-Answer chain using free models"""
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k":3})
    )

    return qa_chain

st.title("PDF Qn and Ans System")
st.write("Upload a PDF and ask questions about its content")

uploaded_file = st.file_uploader("Choose a PDF File", type="pdf")

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size} bytes")

    if st.button("Process PDF"):
        with st.spinner("Extracting text from PDF"):
            text = extract_text_from_pdf(uploaded_file)
            st.success(f"Extracted {len(text)} characters from PDF")

        with st.spinner("Creating embeddings and vector store"):
            st.session_state.vector_store = create_vector_store(text)
            st.success("Vector store created successfully.")

if st.session_state.vector_store is not None:
    st.subheader("Ask Questions")

    question = st.text_input("Enter your question")

    if question and st.button("Get Answer"):
        with st.spinner("Searching answer"):
            qa_chain = setup_qa_chain(st.session_state.vector_store)

            response = qa_chain.run(question)

            st.subheader("Answer:")
            st.write(response)



#-----------------------------------------------------------------------------------------

