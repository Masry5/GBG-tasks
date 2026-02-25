import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document
from collections import defaultdict

load_dotenv()

# --- 1. Setup Models ---
model = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="models/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={
        "tokenizer_kwargs": {"fix_mistral_regex": True}
    },
    encode_kwargs={'normalize_embeddings': True} # Recommended for MiniLM
)

# --- 2. Function to Process CVs (Whole File approach) ---
def process_cvs(directory_path):
    # Load all PDFs (this returns a list where each page is a separate object)
    loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_pages = loader.load()
    
    # Merge pages by filename so one CV = one Document
    combined_content = defaultdict(str)
    for page in raw_pages:
        source_file = page.metadata['source']
        combined_content[source_file] += page.page_content + "\n\n"
    
    # Create the final document list
    final_documents = [
        Document(page_content=text, metadata={"source": source}) 
        for source, text in combined_content.items()
    ]
    
    # Create Vector Store
    vector_store = FAISS.from_documents(final_documents, embeddings)
    return vector_store

# --- 3. Streamlit UI ---
st.title("CV Analysis Assistant 📄")
#st.subheader("One Candidate = One Context")

cv_folder = "CVS" # Ensure this folder exists in your directory

if "vector_db" not in st.session_state:
    if os.path.exists(cv_folder):
        with st.spinner("Indexing full CVs..."):
            st.session_state.vector_db = process_cvs(cv_folder)
            st.success(f"Indexed {len(os.listdir(cv_folder))} CVs fully.")
    else:
        st.error(f"Folder '{cv_folder}' not found!")

# --- 4. Chat Logic ---
prompt = ChatPromptTemplate.from_template("""
You are a professional HR recruiter. Answer the following question using ONLY the provided CV context.
If you find multiple candidates that match, compare them clearly.
If you are asked about candidates' skills, experience, or qualifications, provide a detailed analysis based on the CVs.
If the information is not in the context, say you don't know.
Show detailes of the just the candidates that match the question, do not show the whole CVs.
Don't talk about candidates that do not match the question.

Context:
{context}

Question: {input}
""")

user_question = st.text_input("Ask about your candidates:")

if user_question and "vector_db" in st.session_state:
    # We increase 'k' to ensure it grabs enough full CVs to answer comparative questions
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": user_question})
    
    st.write("### Analysis:")
    st.write(response["answer"])