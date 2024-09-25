# PDF Explainer App

This repository contains the source code for a **PDF Explainer** web application built using **Streamlit**. This app allows users to upload PDF documents, generate summaries, and answer questions based on the content of the PDF using AI-driven capabilities.

## Features
- **PDF Upload**: Users can upload a PDF file to the app.
- **Summarization**: The app generates concise summaries of the uploaded PDF.
- **Question Answering**: Users can ask specific questions about the content of the PDF, and the app provides relevant answers.

## Technology Used
- **Vector Databases**: Pinecone is used
- **RAG**: Modular RAG is used
- **Cohere Embeddings**: Cohere API is used
- **Generative AI**: Gemini Pro model is used with API
- **Streamlit**: Streamlit is used for interface.

## Getting Started

### Prerequisites
Ensure you have the following installed on your machine:
- Python 3.10 
- Virtual environment tool (like `venv` or `conda`)

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/meetgupta7388/Final-year-project.git
cd Final-year-project
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment to isolate dependencies. Here’s how you can do it with venv:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Required Dependencies

Install all required Python packages listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a .env file in the project root directory to store sensitive information such as API keys. Here’s how:

```bash
touch .env
```

In the .env file, add the following

```bash
COHERE_API_KEY = your api key
USER_AGENT="my-app-v1.0"
GEMINI_API_KEY = your api key
PINECONE_API_KEY = your api key
```

### 5. Run the Streamlit App

Once all dependencies are installed, and the environment variables are set up, run the app:

```bash
streamlit run app.py
```

### 7. Demo

For more information and live demo, visit: https://huggingface.co/spaces/utkarsh00010/project_final
