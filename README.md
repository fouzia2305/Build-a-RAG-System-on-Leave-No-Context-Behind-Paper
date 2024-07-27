# RAG-Based QA Streamlit App

Welcome to the RAG-Based QA Streamlit App! This application allows users to interact with a document using Retrieval-Augmented Generation (RAG) techniques. It leverages the Google Gemini API for both generative AI and embeddings, alongside Chroma for document retrieval.

## Features

- **Upload and Process PDFs:** Load and process PDF documents into chunks.
- **Document Embedding:** Use Google Generative AI for creating document embeddings.
- **Vector Store Integration:** Store and retrieve document chunks using Chroma.
- **Question Answering:** Ask questions and receive answers based on the context extracted from the document.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rag-based-qa-streamlit-app.git
    cd rag-based-qa-streamlit-app
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a file named `gemini_api_key.txt` and add your Google Gemini API key.

4. Place your PDF document in the specified path or update the path in the code.

## Configuration

- **Gemini API Key:** Update the path to the `gemini_api_key.txt` file in the script.
- **PDF Path:** Update the path to your PDF document in the script.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the Streamlit app in your web browser. You will see an interface where you can:
   - **Upload a PDF Document** (update code to handle dynamic uploads if needed).
   - **Enter a Question:** Type your question related to the document.
   - **Get Answer:** Click on "Result" to get an answer based on the document context.

## Code Overview

- **Document Loading and Splitting:** Uses `PyPDFLoader` and `NLTKTextSplitter` to process and split the document into chunks.
- **Embedding and Vector Store:** Embeds document chunks using Google Generative AI and stores them using Chroma.
- **Retriever and RAG Chain:** Retrieves relevant document chunks based on user queries and generates answers using a chatbot model.

## Example

Here is a snippet of how to use the app:

```python
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up your Streamlit app, load the PDF, and interact with the model
