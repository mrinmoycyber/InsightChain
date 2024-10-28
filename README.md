# InsightChain 🧠

## Project Goal 🎯
The goal of the InsightChain project is to create an intuitive and interactive chatbot application that enables users to engage in natural language conversations while extracting and processing information from uploaded PDF documents. By leveraging the ChatOllama model (Llama 3.2), InsightChain aims to provide accurate and contextually relevant responses based solely on the content of the PDFs, enhancing user experience and facilitating efficient information retrieval for everyday tasks.

## Features ✨
- **User-Friendly Interface** 🖥️: Intuitive design that allows users to easily interact with the application through a web interface powered by Streamlit.
- **PDF Upload and Processing** 📄: Users can upload multiple PDF documents, which the application processes to extract text content for querying.
- **Conversational AI** 💬: Utilizes the ChatOllama model (Llama 3.2) to provide accurate responses based on the content extracted from uploaded PDFs.
- **Contextual Responses** 📚: Ensures that answers to user queries are derived directly from the uploaded documents, minimizing the risk of hallucination.
- **Real-Time Interaction** ⏱️: Engage in a back-and-forth conversation, asking questions related to the uploaded PDFs or general topics.
- **Chat History** 🗂️: Maintains a history of the conversation, allowing users to review previous interactions and responses.
- **Text Chunking** ✂️: Implements intelligent text chunking to optimize the processing and retrieval of information from lengthy documents.
- **Embedding** 🔍: Leverages embeddings for efficient searching and retrieval of relevant information from the vector store.
- **Session State Management** 🔄: Keeps track of the conversation state, uploaded documents, and chat history to provide a seamless user experience.

## Project Structure 📁
```plaintext
├── app_logic.py
├── app_ui.py 
├── requirements.txt
├── LICENSE
└── README.md
```

## Video Output 🎥

Please click on the link to watch the demo (the file is more than 100 MB, so I can't upload it directly here) - https://drive.google.com/file/d/1zZ1nDZqV7mFTZLLPFPOv3ITJ0qftAcCX/view?usp=sharing

![demo](https://github.com/user-attachments/assets/d4fc24e3-f036-4685-b094-8e06e6d2f8cd)

## Requirements 📦
To run this project, ensure you have the following dependencies installed:

- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain-ollama`
- `faiss-cpu`
- `transformers`
- `torch`

You can install the required packages using pip:

```bash
pip install streamlit PyPDF2 langchain langchain-ollama huggingface-hub
```

Model Setup
Before running the application, you need to install the Ollama CLI and download the Llama model:
- Install the Ollama CLI by following the instructions from the Ollama website.
- After installing the CLI, download the Llama model by running:
```bash
ollama pull llama3.2
```

## Usage 🚀 
Clone the repository:
```bash
git clone https://github.com/mrinmoycyber/InsightChain.git
```
Navigate to the project directory:
```bash
cd InsightChain
```
Run the Streamlit app:
```bash
streamlit run app_ui.py
```




