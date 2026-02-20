import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
#from langchain_community.utilities.sql_database import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
import streamlit as st
#import google.genai as genai
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

load_dotenv()

GOOGLE_API_KEY ="AIzaSyAyCdcsLJcBxW2f39iWB-g-YhA0UTb1rMQ"
DB_URL="postgresql://postgres:XadRqnQWkmenMFTeWltEYAJuEKxLsCYT@switchback.proxy.rlwy.net:36796/railway"

# Initialize LangChain Model
# Note: Using your specified model string
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0
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

def get_sql_from_gemeni(user_query, schema_info):
    # Defining the template to match your original logic exactly
    template = """
    You are an expert PostgresSql Data Analyst.
    
    CRITICAL RULE: The tables were created via pandas. 
    PostgreSQL requires DOUBLE QUOTES around any table or column names that have capital letters.
    Example: Use "Customer" instead of Customer, and "CustomerId" instead of CustomerId.
    
    Here is the database schema:
    {schema_info}
    
    Your task is to write a SQL query that answers the following question:
    {user_query}
    
    - Return ONLY the SQL query.
    - Do NOT include markdown code blocks (```sql).
    - Always use double quotes for table and column names.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # LangChain Chain: Prompt -> Model -> String Output
    chain = prompt | model | StrOutputParser()
    
    response = chain.invoke({
        "schema_info": schema_info,
        "user_query": user_query
    })
    
    # Cleaning as per your original logic
    cleaned_response = response.strip().strip('```sql').strip('```').strip()
    return cleaned_response

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