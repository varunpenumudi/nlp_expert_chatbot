# NLP Expert Chatbot

This is an Expert NLP Chatbot. This Chatbot is trained on about 77,000 NLP Related Papers form arXiv Dataset. Users can ask complex 
questions related to the NLP and it can answer those questions and provide summaries of relavent NLP research Papers from the Dataset. 
This Chatbot Uses LLAMA3.2 Model from Ollama along with Faiss Vector Database for efficeint retrival of related papers. The UI for the
app is built with Streamlit.

The arXiv Dataset(4GB Json file) is filtered using panadas and selected only the NLP Related Papers. There are about 77,000 papers related to NLP Papers
in that Dataset. This filtered dataset is first stored in a csv, feather files and converted to extracted kewords for abstract of each paper using 
nltk techniques. These Keywords along with the titles for all papers are converted to embeddings using the fast embeddings and then these embeddings are 
stored in Faiss index Vector Database.

## Project File Structure
