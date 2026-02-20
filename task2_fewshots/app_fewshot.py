import streamlit as st
import json
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


DB_URL="postgresql://postgres:XadRqnQWkmenMFTeWltEYAJuEKxLsCYT@switchback.proxy.rlwy.net:36796/railway"

# Initialize LangChain Model
# Note: Using your specified model string
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0
)

# 1. Initialize your local model using the LangChain wrapper
# We pass your specific tokenizer_kwargs through model_kwargs
embeddings = HuggingFaceEmbeddings(
    model_name="models/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={
        "tokenizer_kwargs": {"fix_mistral_regex": True}
    },
    encode_kwargs={'normalize_embeddings': True} # Recommended for MiniLM
)


st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with PostgreSQL Database (LangChain version)")

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

def get_schema():
    engine = get_engine()
    inspector_query = text("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            schema_info = {}
            for row in result:
                table_name, column_name, data_type = row
                if table_name not in schema_info:
                    schema_info[table_name] = []
                schema_info[table_name].append(f"{column_name} ({data_type})")
            return schema_info
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return {}


# --- Few-Shot Logic ---

@st.cache_resource
def get_example_selector():
    with open("fewshots.json", "r") as f:
        few_shots = json.load(f)

    # We must tell FAISS to only look at "naturalQuestion" for similarity
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        few_shots,
        embeddings,
        FAISS,
        k=3,
        input_keys=["naturalQuestion"] 
    )
    return example_selector

def get_sql_from_gemeni(user_query, schema_info):
    # 1. THE BRIDGE: Convert your dictionary into a clean string
    # schema_info is the DICT returned by get_schema()
    formatted_schema = ""
    for table_name, columns in schema_info.items():
        formatted_schema += f"\nTable: \"{table_name}\"\n"
        formatted_schema += f"Columns: {', '.join(columns)}\n"

    # 2. Get your cached RAG selector
    example_selector = get_example_selector()
    
    # 3. Define the formatting for your JSON examples
    example_prompt = PromptTemplate(
        input_variables=["naturalQuestion", "sqlQuery"],
        template="User Input: {naturalQuestion}\nSQL Query: {sqlQuery}"
    )

    # 4. Build the final prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="""You are an expert PostgresSql Data Analyst.
    
    CRITICAL RULE: The tables were created via pandas. 
    PostgreSQL requires DOUBLE QUOTES around any table or column names that have capital letters.
    Example: Use "Customer" instead of Customer, and "CustomerId" instead of CustomerId.

    - Return ONLY the SQL query.
    - Do NOT include markdown code blocks (```sql).
    - Always use double quotes for table and column names.
    
    Here is the database schema:
        {schema_info_str} 
        
        Below are relevant examples:""",
        suffix="User Input: {naturalQuestion}\nSQL Query:",
        input_variables=["naturalQuestion", "schema_info_str"]
    )

    # 5. Run the chain
    chain = few_shot_prompt | model | StrOutputParser()
    
    response = chain.invoke({
        "schema_info_str": formatted_schema, # Pass the STRING here, not the DICT
        "naturalQuestion": user_query
    })
    
    return response.strip().strip('```sql').strip('```').strip()


def get_natural_response(question, data):
    template = """
    User Question: {question}
    Data returned from SQL query: {data}
    
    Task: Answer the user's question based on the data returned from the SQL query. If the data does not provide a clear answer, say "The data does not provide a clear answer to the question."
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    
    response = chain.invoke({
        "question": question, 
        "data": data
    })
    return response.strip()

# --- Streamlit UI Logic ---
user_input = st.text_input("Ask a question about your data:")

if user_input:
    schema = get_schema()
    
    with st.spinner("Generating SQL..."):
        sql_query = get_sql_from_gemeni(user_input, schema)
        
    with st.expander("Generated SQL Query"):
        st.code(sql_query, language="sql")
        
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)
            st.dataframe(df)
            
            with st.spinner("Generating final answer..."):
                final_answer = get_natural_response(user_input, df.to_dict(orient="records"))
                st.write(final_answer)
                
    except Exception as e:
        st.error(f"Error executing query: {e}")