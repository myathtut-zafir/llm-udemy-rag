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
    pdf_path="/Users/myathtut/Desktop/Code/llm-udemy-rag/FST20241115.pdf"
    loader=PyPDFLoader(file_path=pdf_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embedding=embeddings)
    vectorstore.save_local('faiss_index_react')
    new_vs_store=FAISS.load_local("faiss_index_react",embeddings=embeddings,allow_dangerous_deserialization=True)
    
    llm= ChatOpenAI()
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrieval_chain=create_retrieval_chain(retriever=new_vs_store.as_retriever(),combine_docs_chain=combine_docs)

    result=retrieval_chain.invoke(input={"input":"ဟူဒိုင်ဗီယာတွင် ပါဝင်သော မွတ်စလင်အရေအတွက် ကို unicode နဲ့ဖော်ပြပါ"})
    print(result["answer"])
    
    
    
    