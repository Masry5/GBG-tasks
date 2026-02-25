import streamlit as st
import os
from dotenv import load_dotenv
from collections import defaultdict

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv()

# --- 1. Setup Models ---
@st.cache_resource # Cache model so it doesn't reload every time
def load_models():
    llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
    embed = HuggingFaceEmbeddings(
        model_name="models/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    return llm, embed

model, embeddings = load_models()

# --- 2. Function to Process CVs (Whole File) ---
@st.cache_resource
def process_cvs(directory_path):
    if not os.path.exists(directory_path):
        return None
    loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_pages = loader.load()
    
    combined_content = defaultdict(str)
    for page in raw_pages:
        source_file = page.metadata['source']
        combined_content[source_file] += page.page_content + "\n\n"
    
    final_documents = [
        Document(page_content=text, metadata={"source": source}) 
        for source, text in combined_content.items()
    ]
    return FAISS.from_documents(final_documents, embeddings)

# --- 3. Streamlit UI State ---
st.set_page_config(page_title="CV Chatbot", layout="wide")
st.title("CV Analysis Chatbot 📄")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Index CVs once
if "vector_db" not in st.session_state:
    cv_folder = "CVS"
    with st.spinner("Processing CV Database..."):
        db = process_cvs(cv_folder)
        if db:
            st.session_state.vector_db = db
            st.success("CVs indexed! You can now ask questions.")
        else:
            st.error(f"Folder '{cv_folder}' not found.")

# --- 4. Chat Logic Setup ---

# This prompt re-writes the user question to be "search-friendly" based on history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Standard Answer Prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional HR recruiter. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Keep the answer detailed but concise.\n\nContext: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the Chain
if "vector_db" in st.session_state:
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 5. The Chat Interface ---

# Display previous messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input for new message
if user_input := st.chat_input("Search CVs (e.g., 'Compare the candidates with Java experience')"):
    
    # Add user message to UI and history
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes..."):
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            
            # Show sources in a small expander
            with st.expander("View Source CVs"):
                sources = {doc.metadata['source'] for doc in response["context"]}
                for s in sources:
                    st.write(f"- {s}")

    # Append to history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer)
    ])