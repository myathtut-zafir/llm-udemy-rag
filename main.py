import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__=='__main__':
    print("Read")
    embedding=OpenAIEmbeddings()
    llm= ChatOpenAI()
    query="What is Pinecone in machine learning?"
    chain=PromptTemplate.from_template(template=query)|llm
    vectorstore=PineconeVectorStore(index_name=os.environ['INDEX_NAME'],embedding=embedding)
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retrieval_chain=create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=combine_docs)
    
    result=retrieval_chain.invoke(input={"input":query})
    print(result)
    
    
    
    