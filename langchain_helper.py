from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from load_vecdb import download_vecdb

# --------------------------------------------------
# LOAD VECTOR DATABASE
# --------------------------------------------------
download_vecdb()
print("Loading Faiss Index")
vectorstore = FAISS.load_local(
    'vecdb_contents', 
    embeddings=FastEmbedEmbeddings(),
    allow_dangerous_deserialization=True
)
print("Loaded Faiss Index")


# --------------------------------------------------
# Function to SEARCH Vector Database
# --------------------------------------------------

def search_vecdb(query: str) -> str:
    docs = vectorstore.search(
        query=query, 
        search_type='similarity'
    )

    res = ''
    for doc in docs:
        if not doc.page_content:
            continue
        res += f"Title: {doc.metadata['title']} \n"
        res += f"Abstract: {doc.metadata['abstract']} \n"
    
    return res


# --------------------------------------------------
# Ollama and Prompt Template
# --------------------------------------------------
llm = ChatOllama(model='llama3.2')
print("LLAMA STARTED.....")

system_prompt = """
You are an intelligent AI  chatbot that serves as an expert in a Natural Language Processing domain, capable of answering complex queries 
and explaining concepts. Use the titles and abstracts of research papers related to user query: {context}. So you are, a chatbot that 
can discuss advanced topics in the NLP field, provide titles, summaries of research papers provided to you nothing more than that, 
and explain complex NLP concepts with concept visualization with understandable examples if possible. Ability to handle follow-up questions on complex topics.
"""

template = ChatPromptTemplate([
    ('system', system_prompt),
    ('placeholder', '{conversation}'),
    ('user', "{query}")
])

chain = template | llm

# --------------------------------------------------
# Get Response to User Query
# --------------------------------------------------
def get_resp(query: str, conversation = []) -> str:
    related_papers = search_vecdb(query=query)
    print(f"Retrived Papers Related to query: {query}")
    resp = chain.invoke({
        "conversation": conversation,
        "context": related_papers,
        "query": query
    })
    print(f"Generated Response to query: {query}")
    return resp.content



if __name__ == "__main__":
    resp = get_resp("Explain Introduction to Arabic Speech Recognition Usin..")
    print(resp)