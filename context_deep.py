import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* Overall Background */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

/* File Uploader Styling */
.stFileUploader {
    background-color: #111111 !important;
    border: 2px solid #00FFAA !important;
    border-radius: 8px;
    padding: 15px;
    color: #FFFFFF !important;
    font-weight: bold;
    text-align: center;
}

/* Upload Button Text */
.stFileUploader label {
    color: #FFFFFF !important;
    font-size: 16px;
}

/* Chat Input Box */
.stChatInput input {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 2px solid #00FFAA !important;
    border-radius: 8px;
    padding: 10px;
}

/* User Chat Message Styling */
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #222222 !important;
    border: 1px solid #00AAFF !important;
    color: #E0E0E0 !important;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

/* Assistant Chat Message Styling */
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #333333 !important;
    border: 1px solid #FFAA00 !important;
    color: #FFFFFF !important;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

/* Avatars */
.stChatMessage .avatar {
    background-color: #00FFAA !important;
    color: #000000 !important;
    font-weight: bold;
    border-radius: 50%;
    padding: 8px;
    display: inline-block;
}

/* Titles and Headers */
h1, h2, h3 {
    color: #00FFAA !important;
    font-weight: bold;
}

/* General Text Fix */
.stChatMessage p, .stChatMessage div {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# --- Constants ---
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the full document context to answer queries concisely. 
If unsure, state that you don't know. Be factual and clear (max 3 sentences).

Query: {user_query} 
Context: {full_document} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# --- Functions ---
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def get_full_document_text(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def generate_answer(user_query, full_text):
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "full_document": full_text})
    
    # Extract response text (Fix)
    if isinstance(response, dict):
        response_text = response.get("text", "⚠️ Error: No response generated.")
    else:
        response_text = str(response)

    # Ensure the response does not exceed 10 sentences
    response_text = limit_sentences(response_text, 10)
    
    return response_text

def limit_sentences(text, max_sentences):
    # Split the text by periods (".") to count sentences
    sentences = re.split(r'(?<=\.)\s+', text)  # Split based on periods and spaces
    # If the response exceeds the maximum sentence count, limit it
    if len(sentences) > max_sentences:
        return ' '.join(sentences[:max_sentences]) + "..."
    return text

# --- UI Layout ---
st.sidebar.title("📌 Instructions")
st.sidebar.info("""
1. Upload a **PDF research document**.
2. Once uploaded, **ask any question** about its content.
3. AI will analyze the **entire document** and provide an **accurate** response.
""")

st.title("📘 InquireAI")
st.markdown("### Your Intelligent Research Assistant 🤖")

uploaded_pdf = st.file_uploader(
    "📂 Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
    st.session_state["document_text"] = ""

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    full_document_text = get_full_document_text(processed_chunks)
    
    # Store document text in session state
    st.session_state["document_text"] = full_document_text

    st.success("✅ Document uploaded successfully! Now, ask your question below.")

# Chat interface for continuous conversation
user_input = st.chat_input("💬 Ask your question about the document...")

if user_input:
    # Add user's question to the conversation history
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("🔍 Analyzing document..."):
        # Generate AI response
        ai_response = generate_answer(user_input, st.session_state["document_text"])

        # Add AI's response to conversation history
        st.session_state["conversation_history"].append({"role": "assistant", "content": ai_response})

        # Display AI's response
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

# Display the entire conversation history
for message in st.session_state["conversation_history"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])
