from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
import pandas as pd
import time


df = pd.read_feather('keywords_extracted/filtered_with_keywords.feather')


# --------------------------------------------------
# CREATE DOCUMENTS
# --------------------------------------------------
start = time.time()
papers = []
for title, abstr, kws in df[['title', 'abstract', 'keywords']].values:
    doc = Document(page_content=title+' '+kws, metadata = {
        "title": title,
        "abstract": abstr,
        "keywords": kws,
    })
    papers.append(doc)
print(f"Took {time.time() - start} Seconds for Documents")


# --------------------------------------------------
# CREATE Database
# --------------------------------------------------
start = time.time()
embeddings = FastEmbedEmbeddings()
vector_store = FAISS.from_documents(
    documents=[Document('')],
    embedding=embeddings,
)

# Add documents in chunks and output that
for i in range(0, 76_000, 1000):
    print(f"{i} documents added to db")
    vector_store.add_documents(
        papers[i:i+1000]
    )
    rem = i+1000

vector_store.add_documents(papers[rem:])

vector_store.save_local('vecdb_contents')
print(f"Took {time.time() - start} Seconds to Save Vector Database")
# Output Took 10833.756065607071 Seconds to Save Vector Database