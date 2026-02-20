import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 1. Setup the LLM (Groq)
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=API_KEY,
    temperature=0
)

# 2. Setup BGE-M3 Embeddings
# BGE-M3 is highly performant for retrieval tasks
encode_kwargs = {'normalize_embeddings': True} # Recommended for BGE models
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have a GPU
    encode_kwargs=encode_kwargs
)

# 3. Load and Split your text file
loader = TextLoader("arabic.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)


# 4. Create Vector Store (FAISS)
vector_db = FAISS.from_documents(chunks, embeddings)

# 5. Build the RAG Chain
# 'stuff' chain takes all retrieved documents and passes them to the prompt
rag_system = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})
)

# 6. Execute Query
question = "What is the main conclusion of the provided text?"
response = rag_system.invoke(question)

print(f"Answer: {response['result']}")


# 6. Execute Query
question = "ما هي القلعة البيضاء"
response = rag_system.invoke(question)

print(f"Answer: {response['result']}")