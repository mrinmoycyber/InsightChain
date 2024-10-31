import streamlit as st
from app_logic import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
    handle_userinput,
    generate_response,
)

# Set up page configuration
st.set_page_config(page_title="InsightChain", page_icon=":books:")

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = []

# Pdf upload
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
    )
    
    if pdf_docs:
        st.session_state.pdf_docs = pdf_docs  # Store uploaded PDFs in session state
    
    if st.button("Process"):
        with st.spinner("Processing PDF documents..."):
            # Get pdf text, split it, and create a vector store
            raw_text = get_pdf_text(st.session_state.pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)

# Chat section
st.title("🧠 InsightChain")
st.write("Ask questions based on uploaded PDFs or have a general chat with InsightChain.")

# User input
with st.form("chat-form"):
    user_question = st.text_input("Enter your question or statement:")
    submit = st.form_submit_button("Submit")

if submit and user_question:
    with st.spinner("Generating response..."):
        if st.session_state.conversation:
            response = handle_userinput(user_question, st.session_state.conversation)
            st.session_state.chat_history.append({"user": user_question, "ollama": response})
        else:
            response = generate_response(user_question)
            st.session_state.chat_history.append({"user": user_question, "ollama": response})

# Chat history
st.write("## Chat History")
for chat in reversed(st.session_state.chat_history):
    if isinstance(chat, dict):
        st.write(f"**🧑 User**: {chat['user']}")
        st.write(f"**🧠 InsightChain**: {chat['ollama']}")
    st.write("---")
