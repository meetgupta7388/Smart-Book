import os
from dotenv import load_dotenv
import streamlit as st
import openai
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up environment variables
USER_AGENT = os.getenv('USER_AGENT', 'my-app-v1.0')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

# Function to read and load PDF document
def read_doc(file):
    file_loader = PyPDFLoader(file)
    documents = file_loader.load()
    return documents

# Function to chunk data
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

# Initialize Cohere embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-english-v2.0",
    user_agent=USER_AGENT
)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("pdf")

# Initialize PineconeVectorStore
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Function to retrieve query results
def retrieve_query(query, k=2):
    matching_results = vector_store.similarity_search(query, k=k)
    return matching_results

# Initialize Google GenerativeAI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to retrieve answers
def retrieve_answers(query):
    doc_search = retrieve_query(query)
    context = "\n".join([doc.page_content for doc in doc_search])
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = model.generate_content(prompt)
    
    return response.text

# Function to handle user input and update chat history
def handle_userinput(user_question):
    answer = retrieve_answers(user_question)
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("AI", answer))

# Function to display chat history
def display_chat_history():
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    
    st.set_page_config(page_title="PDF Explainer App", layout="wide")
    
    css = '''
    <style>
        [data-testid="stSidebar"]{
            min-width: 300px;
            max-width: 300px;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("PDF Explainer App")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload your document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if st.button("Process"):
            if uploaded_file is not None:
                with st.spinner("Processing document..."):
                    # Save the uploaded file temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Read and process the document
                    doc = read_doc("temp.pdf")
                    st.session_state.num_pages = len(doc)
                    
                    documents = chunk_data(doc)
                    st.session_state.num_chunks = len(documents)
                    
                    # Add documents to the vector store
                    vector_store.add_documents(documents)
                    
                    st.success("Document processed successfully!")
                    
                    # Remove the temporary file
                    os.remove("temp.pdf")
            else:
                st.error("Please upload a PDF file first.")

    # Main area for chat history and user input
    display_chat_history()

    user_question = st.chat_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)
        st.rerun()

    # Display some stats
    if 'num_pages' in st.session_state and 'num_chunks' in st.session_state:
        st.sidebar.metric("Number of pages", st.session_state.num_pages)
        st.sidebar.metric("Number of chunks", st.session_state.num_chunks)

if __name__ == '__main__':
    main()