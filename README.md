# **InquireAI: Cache-Augmented Chatbot with DeepSeek R1**

## **Overview**
InquireAI is an **AI-powered chatbot** built using **Cache-Augmented Generation (CAG)** and **DeepSeek R1** to provide **fast, efficient, and intelligent responses** to research-related queries.  
Unlike **Retrieval-Augmented Generation (RAG)**, which retrieves document chunks for each query, CAG stores previously generated responses in a **cache** to **reduce latency** and **optimize computation**.  

### **Key Features**
✅ **Upload a research document (PDF)**  
✅ **Ask questions about its content**  
✅ **Receive instant responses for previously asked queries**  
✅ **Reduce computational overhead by caching results**  

---

## **Tech Stack**
- **LLM**: DeepSeek R1 via **LangChain Ollama**  
- **Framework**: Streamlit  
- **Document Processing**: pdfplumber  
- **Conversational Logic**: LangChain Core & Community  
- **Styling**: Custom Streamlit CSS  

---

## **Screenshots**
### **User Interface**
<img src="https://github.com/user-attachments/assets/0fed6e88-492d-4f8c-9913-3b292b15b7f0" width="600">
<img src="https://github.com/user-attachments/assets/7d24190b-35d0-4f09-8f54-3a871e46d6d0" width="600">

---

## **Installation**

### **Prerequisites**
Ensure you have **Python 3.8+** installed.

### **Clone the Repository**
```bash
git clone https://github.com/your-username/InquireAI.git  
cd InquireAI
