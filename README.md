# NLP Expert Chatbot

This is an Expert NLP Chatbot. This Chatbot is trained on about **77,000 NLP Related Papers** form arXiv Dataset. Users can ask complex 
questions related to the NLP and it can answer those questions and **Provide summaries of relavent NLP research Papers** from the Dataset. 

This Chatbot Uses LLAMA3.2 Model from Ollama along with Faiss Vector Database for efficeint retrival of related papers. The UI for the
I have deployed this app on the Streamlit Cloud, and you can try that by [Clicking Here.](https://nlpexpertbot.streamlit.app/)
app is built with Streamlit. 

The arXiv Dataset(4GB Json file) is **Filtered using panadas** and selected only the NLP Related Papers. There are about 77,000 papers related to NLP Papers
in that Dataset. This filtered dataset is first **Stored in a CSV, feather files** and **Extracted Keywords** for abstract of each paper using 
nltk techniques. These Keywords along with the titles for all papers are **Converted to embeddings using the Fast embeddings library** and then these embeddings are stored in Faiss index Vector Database.


## Project File Structure

The file Structure of this whole project is shown below

```python
│  
├── chatbot.py                         (Code for Streamlit chatbot)
├── langchain_helper.py                (Helper Code to Langchain LLM Usage)
├── README.md                          (README File)
├── store_vecdb.py                     (Code to Store Filtered Dataset in Vector Database) 
├── load_vecdb.py                      (Code to Download the Vector Database Stored on Kaggle)
│
|
├───filtered_data                     (Filtered Dataset folder)
│       all_categories.json           (All Categories in arXiv Dataset)
│       dataset_filtering.ipynb       (Code to filtering arXiv Dataset for NLP Papers )
│       filtered_data.csv             (Filtered Dataset in CSV Format)
│       filtered_data.feather         (Filtered Dataset in Feather Format)
│
|
└───keywords_extracted                  (Folder for keywords of all files)P
        abstract_keywords.pickle        (Keywords in pickle file)
        filtered_with_keywords.feather  (Filtered Dataset with Keywords in Feather format)
        keywords_extract.py             (Code for Keywords extraction)
```

## How to run the app

To run this streamlit chatbot do the following steps in your computer.

1. Install the [Ollama](https://ollama.com/download) on your computer, and run the following command to install llama3.2 llm.

    ```bash
    ollama pull llama3.2
    ```

2. Clone the repository

    ```bash
    git clone https://github.com/varunpenumudi/nlp_expert_chatbot
    ```

3. `cd` into the repository folder

    ```bash
    cd nlp_expert_chatbot
    ```

4. Install the python requirements

    ```bash
    pip install -r requirements.txt
    ```

5. Run the streamlit app 

    ```bash
    streamlit run chatbot.py
    ```

## How it Works

The NLP Expert Chatbot operates through the following key stages:

1. **Dataset Filtering:** The arXiv dataset (a large 4GB JSON file) is processed to extract approximately 77,000 papers relevant to Natural Language Processing (NLP). This filtering was performed using Pandas for efficient data manipulation.
2. **Data Preprocessing and Feature Engineering:**
    * The filtered dataset is stored in CSV and Feather formats for efficient access and analysis.
    * Keywords are extracted from the abstract of each paper using NLTK (Natural Language Toolkit) for NLP techniques.
    * Paper titles and extracted abstract keywords are used to generate vector embeddings using fastText embeddings.
3. **Vector Database Indexing:**  These embeddings are then indexed and stored in a Faiss vector database. Faiss enables fast and efficient similarity search for relevant papers based on user queries.
4. **Chatbot Interaction with LLM:**
    * The Streamlit UI provides an interactive chat interface for users to ask NLP-related questions.
    * User queries are processed to find semantically similar research papers from the Faiss vector database.
    * The LLAMA3.2 language model (accessed via Ollama) is used to generate answers and explanations, leveraging the retrieved research paper content as context.
5. **Explanation and Summarization:** The chatbot can provide:
    * Answers to complex NLP questions based on its training data and retrieved research papers.
    * Summaries of relevant research papers to give users a quick overview of the content.

## Key Features

* **Expertise in Natural Language Processing:** Trained on a large corpus of NLP research papers from arXiv.
* **Complex Question Answering:** Capable of answering intricate questions related to NLP concepts, techniques, and research trends.
* **Research Paper Summarization:** Provides concise summaries of relevant arXiv papers.
* **Concept Explanation:** Explains complex NLP concepts in an understandable manner.
* **Follow-up Question Handling:**  Maintains conversation context to handle follow-up questions effectively.
* **Paper Searching:**  Allows users to search for research papers by keywords via the Streamlit UI.
* **Powered by Open Source Technologies:** Built using LLAMA3.2 (Ollama), Faiss, NLTK, and Streamlit.

These are all the key features of this NLP Expert Chabot. In summary, this chatbot can explain you about the summaries of NLP Papers related to your query and with examples if needed and can answer the user's follow up questions.