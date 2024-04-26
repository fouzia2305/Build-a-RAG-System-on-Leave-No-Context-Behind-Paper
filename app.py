from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st 

# Title and header for the Streamlit app
st.title("RAG-Based QA")
st.subheader("Exploring 'Leave No Context Behind' Paper 📄🔍")



# Gemini-api-key
f = open(r"C:\Users\FOUZIA KOUSER\OneDrive\Desktop\RAG2\gemini_api_key.txt")
GEMINI_API_KEY = f.read()

# Create the chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-1.5-pro-latest")

# Load a document

pdf_path = r"C:\Users\FOUZIA KOUSER\OneDrive\Desktop\2404.07143v1.pdf"

# Create a PyPDFLoader instance
loader = PyPDFLoader(pdf_path)

# Load and split the document
data = loader.load_and_split()

# Spliting the document into chunks

from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(data)


# # Create the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model="models/embedding-001")



# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_rag")

# Persist the database on drive
db.persist()
# Set up a connection with the Chroma for retrieval
connection = Chroma(persist_directory="./Chroma_db_rag", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = connection.as_retriever(search_kwargs={"k": 5})

# User query
user_query = st.text_input("Enter your question:")

# Chatbot prompt templates
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."),
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext: {Context}\nQuestion: {question}\nAnswer:")
])

# Output parser for chatbot response
output_parser = StrOutputParser()

# Function to format retrieved documents
def format_docs(docs):
    formatted_content = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
    return formatted_content if formatted_content else "No relevant context found."

# RAG chain for chatbot interaction
rag_chain = (
    {"Context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

if st.button("Result"):
    if user_query:
        response = rag_chain.invoke(user_query)
        st.write("Answer:")
        st.write(response)