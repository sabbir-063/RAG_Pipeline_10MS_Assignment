# RAG-Based Chatbot

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python. The chatbot leverages vector embeddings and a retrieval system to provide contextually relevant responses.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) virtualenv for virtual environment management

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv rag_chatbot_env

# Activate the environment
# On Windows:
rag_chatbot_env\Scripts\activate
# On Unix or MacOS:
source rag_chatbot_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root directory with your API keys and configurations:
```bash
HUGGINGFACE_TOKEN=huggingface_key
```

### 4. Run the FastAPI Application
```bash
uvicorn main:app --reload
```

The application will be available at:
- Local: http://127.0.0.1:8000

## File Structure

```bash
├── config/
│ ├── constants.py # Project constants
├── data/
│ ├── bangla/ # Bangla documents
│ ├── english/ # English documents
│── app.py  #Fastapi app run
│── data_processing.py # Preprocess documents
│── rag_system.py # main RAG pipeline
├── requirements.txt # Python dependencies
├── README.md # This file
└── .env # Environment variables
```
