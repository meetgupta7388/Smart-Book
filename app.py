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
import json
from langchain_community.document_loaders import PyPDFLoader 

# Load environment variables
load_dotenv()

# Set the page configuration here (this will be applied to the entire app)
st.set_page_config(page_title="PDF and Quiz App", layout="wide")

# Set up environment variables
USER_AGENT = os.getenv('USER_AGENT', 'my-app-v1.0')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
#GOOGLE_KEY = os.getenv('Google_KEY')

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

# Configure Gemini API with the API key
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Error in configuring Generative AI: {str(e)}")

# Load the Generative AI model
try:
    model_1 = genai.GenerativeModel("gemini-1.5-pro-001")
except Exception as e:
    st.error(f"Error in loading the Generative AI model: {str(e)}")

# Set Streamlit page configurations
#st.set_page_config(page_title="MCQ Generator", layout="wide")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    try:
        # Save the uploaded PDF to a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())  # Get the file buffer and write it to disk
            
        # Load the PDF using PyPDFLoader
        file_loader = PyPDFLoader("temp.pdf")
        documents = file_loader.load()
        # Join the documents into a single text block
        text_content = " ".join([doc.page_content for doc in documents])
        return text_content
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

# Function to fetch questions using Gemini API
@st.cache_data
def fetch_questions(text_content, number,topic, quiz_level):
    try:
        RESPONSE_JSON = {
      "mcqs" : [
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        },
        {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        },
        {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct choice option in the form of a, b, c or d",
        }
      ]
    }

        prompt_template=f"""
    Text: {text_content}
    You are an expert in generating MCQ type quiz on the basis of provided content. 
    Given the above text, create a quiz of {number} multiple choice questions on {topic} topic keeping difficulty level as {quiz_level}. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide.
    Ensure to make an array of {number} MCQs referring the following response json.
    Here is the RESPONSE_JSON: 

    {RESPONSE_JSON}

    """
        # Call the Gemini API with the formatted prompt
        response = model_1.generate_content([prompt_template, text_content])
        # Extract the JSON string from the response
        response_content = response.candidates[0].content.parts[0].text
        # Clean up the JSON string
        json_content = response_content.strip("```json\n").strip("```").strip()  # Clean the formatting
        # Replace single quotes with double quotes
        json_content = json_content.replace("'", '"')
        # Try to load the JSON and handle potential errors
        try:
            parsed_json = json.loads(json_content)
        except json.JSONDecodeError as e:
            st.error(f"JSON Decode Error: {e}")
            return []

        return parsed_json.get("mcqs", [])
    
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []


# Main function to run the Streamlit app
def main():
    load_dotenv()
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'login'  # default page

    # Function to switch pages
    def switch_page(page_name):
        st.session_state.page = page_name
        st.rerun()

    if st.session_state.page == 'login':
        st.title("Log In Page")
        username = st.text_input("Enter username")
        password = st.text_input("Enter Password", type="password")
        button = st.button("Log In")

        # Check for valid credentials and switch page
        if button:
            if username == "bhanumota" and password == "chemistry99":
                switch_page('second')
            else:
                st.error("Invalid username or password")
    elif st.session_state.page == 'second':

        # Tabs for Chat and Test
        tab1, tab2 = st.tabs(["Chat", "Test"])
        
        # "Chat" Tab
        with tab1:
            st.header("PDF Explainer App")

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
                if st.button("Logout"):
                   switch_page('login')

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

        # "Test" Tab for Quiz Generation
        with tab2:
            st.title("Quiz Generator App")

        # File uploader for PDF
            uploaded_pdf = uploaded_file

            if uploaded_pdf is not None:
                text_content = extract_text_from_pdf(uploaded_pdf)
                
                if text_content:
                    # Select number of MCQs and quiz level
                    number = st.number_input("Enter the number of MCQs you want to practice", min_value=1, max_value=10, value=3)
                    topic = st.text_input("Enter the topic from pdf.")
                    quiz_level = st.selectbox("Select quiz level:", ["Easy", "Medium", "Hard"])
                    quiz_level_lower = quiz_level.lower()
                    
                    # Initialize session_state for quiz generation tracking
                    session_state = st.session_state
                    if 'quiz_generated' not in session_state:
                        session_state.quiz_generated = False
                    
                    if not session_state.quiz_generated:
                        session_state.quiz_generated = st.button("Generate Quiz")
                    
                    if session_state.quiz_generated:
                        questions = fetch_questions(text_content=text_content, number=number,topic = topic, quiz_level=quiz_level)
                        
                        if questions:
                            selected_options = []
                            correct_answers = []
                            for question in questions:
                                options = list(question["options"].values())
                                selected_option = st.radio(question["mcq"], options)
                                selected_options.append(selected_option)
                                correct_answers.append(question["options"][question["correct"]])
                            
                            # Submit button
                            if st.button("Submit"):
                                marks = 0
                                st.header("Quiz Result:")
                                for i, question in enumerate(questions):
                                    selected_option = selected_options[i]
                                    correct_option = correct_answers[i]
                                    st.subheader(f"{question['mcq']}")
                                    st.write(f"You selected: {selected_option}")
                                    st.write(f"Correct answer: {correct_option}")
                                    if selected_option == correct_option:
                                        marks += 1
                                st.subheader(f"You scored {marks} out of {len(questions)}")
                else:
                    st.error("Error processing PDF content.")


if __name__ == '__main__':
    main()
