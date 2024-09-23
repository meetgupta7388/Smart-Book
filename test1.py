import streamlit as st
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader  # Updated import

# Load environment variables
load_dotenv()

# Configure Gemini API with the API key
try:
    genai.configure(api_key="AIzaSyDsigaU8iA_UYBLZSLrkbpv9J6sabJjH2g")
except Exception as e:
    st.error(f"Error in configuring Generative AI: {str(e)}")

# Load the Generative AI model
try:
    model = genai.GenerativeModel("gemini-1.5-pro-001")
except Exception as e:
    st.error(f"Error in loading the Generative AI model: {str(e)}")

# Set Streamlit page configurations
st.set_page_config(page_title="MCQ Generator", layout="wide")

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
    Ensure to make an array of 3 MCQs referring the following response json.
    Here is the RESPONSE_JSON: 

    {RESPONSE_JSON}

    """
        # Call the Gemini API with the formatted prompt
        response = model.generate_content([prompt_template, text_content])
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

def main():
    st.title("Quiz Generator App")

    # File uploader for PDF
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

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
                questions = fetch_questions(text_content=text_content, number=number,topic = topic, quiz_level=quiz_level_lower)
                
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

if __name__ == "__main__":
    main()
