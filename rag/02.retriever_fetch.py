import os

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# set db path
current_dir = os.path.dirname(os.path.abspath(__file__))
peristent_dir = os.path.join(current_dir, "db",'chroma_db')

# set embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# point to local vector db
db = Chroma(
    persist_directory=peristent_dir,
    embedding_function=embeddings
)

# set db as retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)


query = "Who is Odysseus's wife?"

# fetch query
relavent_docs = retriever.invoke(query)

for i in range(len(relavent_docs)):
    print(f"Doc num {i+1} : {relavent_docs[i]}")
