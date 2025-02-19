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
You are an expert research assistant. Answer queries concisely using the provided document context.
If unsure, state that you don't know. Be factual and clear (max 10 sentences).

Query: {user_query} 
Context: {full_document} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# --- Initialize Cache ---
if "response_cache" not in st.session_state:
    st.session_state["response_cache"] = {}  # Dictionary to store past Q&A pairs
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "document_text" not in st.session_state:
    st.session_state["document_text"] = ""

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
    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_processor.split_documents(raw_documents)

def get_full_document_text(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def generate_answer(user_query, full_text):
    """ Generate response using cache if available; otherwise, call LLM """
    # Check if the response is already in the cache
    if user_query in st.session_state["response_cache"]:
        return st.session_state["response_cache"][user_query]  # Return cached response

    # If not cached, generate response
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "full_document": full_text})

    # Extract response text
    if isinstance(response, dict):
        response_text = response.get("text", "⚠️ Error: No response generated.")
    else:
        response_text = str(response)

    # Limit response length and store in cache
    response_text = limit_sentences(response_text, 10)
    st.session_state["response_cache"][user_query] = response_text  # Cache the response

    return response_text

def limit_sentences(text, max_sentences):
    sentences = re.split(r'(?<=\.)\s+', text)
    return ' '.join(sentences[:max_sentences]) + ("..." if len(sentences) > max_sentences else "")

# --- UI Layout ---
st.sidebar.title("📌 Instructions")
st.sidebar.info("""
1. Upload a **PDF research document**.
2. Once uploaded, **ask any question** about its content.
3. The chatbot will first check its **cache** to provide faster responses.
4. If no answer is cached, the AI will generate a new response and store it for future use.
""")

st.title("📘 InquireAI (Cache-Augmented)")
st.markdown("### Your Fast and Efficient Research Assistant 🚀")

uploaded_pdf = st.file_uploader("📂 Upload a Research Document (PDF)", type="pdf")

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    full_document_text = get_full_document_text(processed_chunks)
    st.session_state["document_text"] = full_document_text
    st.success("✅ Document uploaded successfully! Ask questions below.")

# Chat interface for continuous conversation
user_input = st.chat_input("💬 Ask your question about the document...")

if user_input:
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("🔍 Checking cache..."):
        ai_response = generate_answer(user_input, st.session_state["document_text"])

    st.session_state["conversation_history"].append({"role": "assistant", "content": ai_response})

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)

# Display the entire conversation history
for message in st.session_state["conversation_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
