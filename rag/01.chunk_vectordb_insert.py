from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

#set paths for vector_db locally
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", 'chroma_db')
print(persistent_directory)

doc_path = "/Users/shakibibnashameem/Documents/Practice/LangChain/rag/odessy.txt"


if not os.path.exists(persistent_directory):

    print('---SYS MSG---: Initializing...')
    #load doc
    loader = TextLoader(doc_path)
    doc = loader.load()
    print('---SYS MSG---: Document Loaded')

    #split and chunk
    text_splitter = CharacterTextSplitter(chunk_size = 1500, chunk_overlap = 100)
    chunks = text_splitter.split_documents(doc)
    print('---SYS MSG---: Chunking Done')

    #create embeddings
    embedding = OllamaEmbeddings(
        model='nomic-embed-text'
    )
    print('---SYS MSG---: Embeddings Created')

    #insert into vectordb
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persistent_directory
    )
    print('---SYS MSG---: Insert done')

    
else:
    print("Vector store already exists")