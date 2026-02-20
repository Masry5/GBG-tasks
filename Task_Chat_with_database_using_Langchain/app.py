import streamlit as st
import google.generativeai as genai
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
#from typer import prompt

GOOGLE_API_KEY ="AIzaSyAZWfesszIz-xfK67JQPsUeD6hED0jJhs0"

DB_URL="postgresql://postgres:XadRqnQWkmenMFTeWltEYAJuEKxLsCYT@switchback.proxy.rlwy.net:36796/railway"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")


st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with PostgressQL Database")

st.cache_resource
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
    prompt= f"""
    You are an expert PostgresSql Data Analyst.
    
    Here is the database schema:
    {schema_info}
    
    Your task is to write a SQL query that answers the following question:
    {user_query}
    
    The tables were created via pandas.
    - If columns or tables names are MixedCase, use double quotes around them in the SQL query.
    - Return Only the SqL query, without any explanation or comments.
    """
    response = model.generate_content(prompt)
    #return response.text.strip()
    cleaned_response = response.text.strip().strip('```sql').strip('```')
    return cleaned_response



def get_natural_response(question, data):
    prompt=f"""
    User Question: {question}
    Data returned from SQL query: {data}
    
    Task: Answer the user's question based on the data returned from the SQL query. If the data does not provide a clear answer, say "The data does not provide a clear answer to the question."
    """
 
    response = model.generate_content(prompt)
    return response.text.strip()


