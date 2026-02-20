# 📊 Semantic SQL Chatbot: AI-Powered Postgres Analyst

An intelligent database assistant built with LangChain, Google Gemini, and Streamlit. This application allows users to query a PostgreSQL database using natural language. It leverages a Few-Shot RAG approach to retrieve relevant SQL examples from a local vector store, ensuring highly accurate query generation even for complex schemas.

---

## 🌟 Key Features

- **Natural Language to SQL**: Converts plain English questions into optimized PostgreSQL queries.
- **Dynamic Few-Shot Learning**: Uses a FAISS vector store to retrieve the most relevant SQL examples for every user question.
- **Schema-Aware Generation**: Automatically fetches and formats your database schema to provide context to the LLM.
- **Local Embeddings**: Utilizes MiniLM-L12-v2 locally via HuggingFace for fast, private semantic search.
- **Interactive UI**: A clean Streamlit interface with expandable SQL code blocks and data previews.
- **Data-to-Answer**: Executes the generated SQL and provides a human-readable summary of the results.

---

## 🏗️ Technical Stack

- **Orchestration**: LangChain (LCEL)  
- **LLM**: Google Gemini (`gemini-2.5-flash`)  
- **Vector Store**: FAISS (Facebook AI Similarity Search)  
- **Embeddings**: HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)  
- **Frontend**: Streamlit  
- **Database**: PostgreSQL via SQLAlchemy  

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+  
- A running PostgreSQL database  
- A Google AI (Gemini) API Key  

---

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/sql-chatbot.git
cd sql-chatbot
pip install streamlit pandas sqlalchemy psycopg2-binary langchain-google-genai langchain-community langchain-huggingface faiss-cpu python-dotenv
```

---

### 3. Environment Setup

Create a `.env` file in the root directory and add your credentials:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/your_database
```

> **Note:** Avoid hardcoding sensitive credentials inside the application. Always use environment variables in production.

---

### 4. Prepare Few-Shot Examples

Create a `fewshots.json` file to help the model learn your specific SQL patterns:

```json
[
  {
    "naturalQuestion": "How many customers are in the USA?",
    "sqlQuery": "SELECT COUNT(*) AS TotalCustomers FROM \"Customers\" WHERE \"Country\" = 'USA';"
  },
  {
    "naturalQuestion": "What is the total revenue for 2023?",
    "sqlQuery": "SELECT SUM(\"Amount\") AS TotalRevenue FROM \"Orders\" WHERE EXTRACT(YEAR FROM \"OrderDate\") = 2023;"
  }
]
```

These examples are embedded and stored in FAISS to enable semantic retrieval.

---

### 5. Running the App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 💡 How It Works

### 1️⃣ Schema Retrieval  
The app queries `information_schema` to dynamically extract:
- Table names  
- Column names  
- Data types  

This ensures the LLM always works with up-to-date database structure.

---

### 2️⃣ Semantic Search  
- The user question is converted into embeddings using MiniLM.
- FAISS compares it against stored few-shot examples.
- The most relevant examples are selected.

---

### 3️⃣ Prompt Construction  
A `FewShotPromptTemplate` combines:
- Database schema  
- Retrieved SQL examples  
- User question  

This structured context dramatically improves SQL accuracy.

---

### 4️⃣ SQL Generation  
Google Gemini generates a PostgreSQL-compatible query while:
- Respecting case sensitivity  
- Using double quotes for capitalized identifiers  
- Matching correct column data types  

---

### 5️⃣ Execution & Answer Synthesis  
- The generated SQL query is executed via SQLAlchemy.
- Results are loaded into a Pandas DataFrame.
- The DataFrame is passed back to Gemini.
- A clean, human-readable explanation is generated.

---

## 🛡️ Best Practices Included

- **Resource Caching**: Uses `@st.cache_resource` for database engine and vector store caching.
- **Type-Safe Prompting**: Schema includes data types to prevent invalid comparisons.
- **PostgreSQL Compliance**: Enforces proper quoting for identifiers.
- **Separation of Secrets**: Encourages `.env` usage for secure deployments.
- **Modular Design**: Clear separation between retrieval, generation, execution, and UI layers.
