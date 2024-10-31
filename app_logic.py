from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama

# Define pdf text extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function for text chunking for vector store
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

# Function for vector store for pdf text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434/")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

# User input
def handle_userinput(user_question, conversation):
    prompt = (
        "You are an AI assistant trained to provide information based only on the content extracted from the uploaded PDF document. "
        "Your responses should directly reference information from the document and avoid any assumptions, generalizations, or information not contained within it. "
        "When answering the following question, ensure your response is grounded in the specific text of the PDF, and refrain from providing any additional context or external knowledge."
    )
    response = conversation({'question': f"{prompt}\n{user_question}"})
    return response['answer']

# Function for chat with Ollama
def generate_response(input_text):
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434/")
    response = model.invoke(input_text)
    return response.content
