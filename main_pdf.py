import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
if __name__=='__main__':
    print("hello")
    pdf_path="/Users/myathtut/Desktop/Code/llm-udemy-rag/2210.03629v3.pdf"
    loader=PyPDFLoader(file_path=pdf_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embedding=embeddings)
    
    
    