import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter




load_dotenv()
if __name__=='__main__':
    print("Hello")
    loader=TextLoader("/Users/myathtut/Desktop/Code/llm-udemy-rag/mediumblog1.txt")
    document=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    # print(f"created{len(texts)}")
    embeddings=OpenAIEmbeddings()
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ['INDEX_NAME'])
    
    
