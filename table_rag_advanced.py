import streamlit as st
import pandas as pd
import os
import json
import uuid
import time
from sqlalchemy import create_engine, text

# Import LLM and prompt classes
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated

# For SQL Database utilities
from langchain_community.utilities import SQLDatabase

# Use st-aggrid for interactive table display
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except ImportError:
    st.error("Please install st-aggrid (pip install streamlit-aggrid) to view results in an interactive table.")
    st.stop()

######################################
# SESSION SETUP: Unique DB per User Session
######################################

# Create a temporary directory for session databases.
TEMP_DB_DIR = "temp_db"
if not os.path.exists(TEMP_DB_DIR):
    os.makedirs(TEMP_DB_DIR)

# Cleanup function to remove old session databases (e.g., older than 1 hour).
def cleanup_old_session_dbs(folder, expiration_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            mtime = os.path.getmtime(filepath)
            if now - mtime > expiration_seconds:
                os.remove(filepath)

cleanup_old_session_dbs(TEMP_DB_DIR)

# Create a unique session ID for this user session (if not already set).
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]
db_file = os.path.join(TEMP_DB_DIR, f"training_{session_id}.db")
st.session_state["db_file"] = db_file

######################################
# 1. GENERALIZED FILE INPUT & DATA LOAD
######################################

st.set_page_config(layout="wide")
st.title("TableAgent")
st.markdown(
    """
    Upload a CSV, XLSX, or JSON file. The app will ingest the data, build a database, and allow you to ask SQL-based questions.
    """
)

# Sidebar inputs for configuring the LLM
st.sidebar.header("LLM Configuration")
base_url = st.sidebar.text_input("Base URL", "http://127.0.0.1:1234/v1")
api_key = st.sidebar.text_input("API Key", os.environ.get("OPENAI_API_KEY", "lm-studio"))
model = st.sidebar.text_input("Model", "qwen2.5-coder-7b-instruct")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

# New toggleable options:
show_sql_debug = st.sidebar.checkbox("Show SQL Query for debugging", value=False)
tell_me_about_data = st.sidebar.checkbox("Tell me about the data", value=False)
explain_query_toggle = st.sidebar.checkbox("Explain query", value=False)

# Sidebar multiselect for extra columns that should always be included.
extra_columns = st.sidebar.multiselect(
    "Always display these columns (if applicable)", 
    options=[],  # This will be set after file upload.
    default=[]
)

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json"])
if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Replace spaces in column names with underscores
        df.columns = [col.replace(" ", "_") for col in df.columns]
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            st.error("Duplicate column names detected after processing. Please check the file.")
            st.stop()
        
        # Store the dataframe in session state so that it persists for this user.
        st.session_state["df"] = df

        # Update the extra_columns multiselect options (if not already set)
        extra_columns = st.sidebar.multiselect(
            "Always display these columns (if applicable)", 
            options=df.columns.tolist(), 
            default=[]
        )

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    if "df" in st.session_state:
        df = st.session_state["df"]
        # Update the extra_columns multiselect options (if not already set)
        extra_columns = st.sidebar.multiselect(
            "Always display these columns (if applicable)", 
            options=df.columns.tolist(), 
            default=[]
        )
    else:
        st.info("Please upload a CSV, XLSX, or JSON file to proceed.")
        st.stop()

######################################
# 2. SET UP THE LLM CONNECTION AND DATABASE ENGINES
######################################

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model,
    temperature=temperature,
)

# Create a full-access engine (used for schema introspection)
if "full_engine" not in st.session_state:
    full_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    st.session_state["full_engine"] = full_engine
else:
    full_engine = st.session_state["full_engine"]

# Write (or replace) the table (always named 'training' for simplicity)
df.to_sql("training", full_engine, if_exists="replace", index=False)

# Create a read-only engine for query execution.
if "ro_engine" not in st.session_state:
    ro_engine = create_engine(
        f"sqlite:///file:{db_file}?mode=ro&uri=true",
        connect_args={"check_same_thread": False}
    )
    st.session_state["ro_engine"] = ro_engine
else:
    ro_engine = st.session_state["ro_engine"]

# Instantiate SQLDatabase objects:
db_for_schema = SQLDatabase(engine=full_engine, sample_rows_in_table_info=5)
schema_info = db_for_schema.get_table_info()  # Detailed schema info
db = SQLDatabase(engine=ro_engine)

######################################
# 3. (OPTIONAL) DATA OVERVIEW: "TELL ME ABOUT THE DATA"
######################################

if tell_me_about_data:
    st.subheader("Data Overview")
    # Display a random sample of 5 rows using st-aggrid
    sample_df = df.sample(n=5) if len(df) >= 5 else df
    gb_sample = GridOptionsBuilder.from_dataframe(sample_df)
    gb_sample.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
    st.markdown("#### Random Sample of 5 rows:")
    AgGrid(
        sample_df,
        gridOptions=gb_sample.build(),
        height=200,
        fit_columns_on_grid_load=True,
        key="aggrid_sample",
        update_mode=GridUpdateMode.NO_UPDATE
    )
    
    # Use the table schema and sample rows to generate an explanation of the columns.
    st.markdown("#### Explanation of the Data Columns")
    explanation_prompt = f"""
You are an expert data analyst. Below is the table schema and a sample of rows from the table.
Please provide the customer with ONLY the **clear and concise explanation** of each column.

Table Schema:
{schema_info}

Sample Rows:
{sample_df.to_json(orient="records", indent=2)}

<clear and concise explanation>:
"""
    data_explanation = llm.invoke(explanation_prompt)
    st.write(data_explanation)

######################################
# 4. INITIALIZE PROMPT TEMPLATE
######################################

prompt_template_text = '''
You are a top tier data analysis algorithm that strives to understand the customer's needs and provide the most relevant data.   

Given the customer's input question, generate a syntactically correct {dialect} SQL query that retrieves only the necessary columns to answer the question. The query must include, in its SELECT clause, every column that appears in its WHERE clause. Do not include a LIMIT clause in the query. You may order the results by a relevant column to highlight the most interesting examples.

The output must be formatted as JSON with a single field "query". For example:

{{ 
  "query": "SELECT COUNT(*) FROM Database;"
}}

Pay careful attention to use **only** the column names that you can see in the **schema_info**. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

<schema_info>
{table_info}
</schema_info>

**Extra Instruction:** If applicable (i.e. if the query is not an aggregation query), always include the following extra_columns in the SELECT clause: 

<extra_columns> 
{extra_columns} 
</extra_columns>. 

If the query involves an aggregate function (e.g. COUNT, MIN, MAX, AVG, SUM), ignore this instruction.

Question: {input}
'''

query_prompt_template = ChatPromptTemplate.from_template(prompt_template_text)

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

######################################
# 5. SQL CHAIN FUNCTIONS
######################################

def generate_sql_query(question: str) -> dict:
    """Generate a SQL query based on the user question, schema info, and extra columns."""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "table_info": schema_info,
        "input": question,
        "extra_columns": ", ".join(extra_columns) if extra_columns else "None"
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    if show_sql_debug:
        st.subheader("Generated SQL Query")
        st.code(result["query"], language="sql")
    return {"query": result["query"]}

def execute_sql_query(state: dict) -> dict:
    """Execute the generated SQL query in a read-only manner and return row count and results."""
    query = state["query"]
    if not query.strip().lower().startswith("select"):
        return {"result": "Error: Only SELECT queries are permitted.", "row_count": 0}
    with ro_engine.connect() as conn:
        result = conn.execute(text(query))
        rows = [dict(row._mapping) for row in result]
        row_count = len(rows)
    return {"result": json.dumps(rows, indent=2), "row_count": row_count}

def explain_generated_query(query: str) -> str:
    """Generate an explanation of what the provided SQL query does."""
    explanation_prompt = f"""
Explain in plain English what the following SQL query does, including what data it retrieves, which conditions it applies, and any ordering of the results:

{query}

Explanation:
"""
    explanation = llm.invoke(explanation_prompt)
    return explanation

######################################
# 6. STREAMLIT USER INTERFACE FOR QA
######################################

st.markdown("### Ask a question about your data:")
user_question = st.text_input("Your question:")

if st.button("Submit") and user_question:
    st.write("Processing your question...")
    sql_state = {"question": user_question}
    sql_state.update(generate_sql_query(user_question))
    sql_state.update(execute_sql_query(sql_state))
    
    # Convert the JSON string of results to a DataFrame.
    try:
        df_result = pd.read_json(sql_state["result"])
    except Exception as e:
        st.error(f"Error parsing SQL results: {e}")
        df_result = pd.DataFrame()
    
    st.subheader("Results")
    if not df_result.empty:
        # Use AgGrid to display an interactive, resizable, and wrapped table.
        gb = GridOptionsBuilder.from_dataframe(df_result)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gridOptions = gb.build()
        AgGrid(
            df_result, 
            gridOptions=gridOptions, 
            height=300, 
            fit_columns_on_grid_load=True, 
            key="aggrid_results", 
            update_mode=GridUpdateMode.NO_UPDATE
        )
    else:
        st.text("No results returned.")
    
    # (OPTIONAL) Explain the SQL query if the toggle is enabled.
    if explain_query_toggle and "query" in sql_state:
        st.subheader("Query Explanation")
        query_explanation = explain_generated_query(sql_state["query"])
        st.write(query_explanation)
