import streamlit as st
import pandas as pd
import os
import json
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
# 1. GENERALIZED FILE INPUT & DATA LOAD
######################################

st.set_page_config(layout="wide")
st.title("TableAgent")
st.markdown(
    """
    Upload a CSV, XLSX, or JSON file. The app will ingest the data, build a database, and allow you to ask SQL-based questions.
    """
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
        
        #st.success("File uploaded and processed successfully!")
        #st.write(df.head())
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            st.error("Duplicate column names detected after processing. Please check the file.")
            st.stop()

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV, XLSX, or JSON file to proceed.")
    st.stop()

######################################
# 2. SET UP THE DATABASE ENGINES
######################################

# Create a full-access engine (used for schema introspection)
full_engine = create_engine("sqlite:///training.db", connect_args={"check_same_thread": False})
# Write (or replace) the table (always named 'training' for simplicity)
df.to_sql("training", full_engine, if_exists="replace", index=False)

# Create a read-only engine for query execution.
ro_engine = create_engine(
    "sqlite:///file:training.db?mode=ro&uri=true",
    connect_args={"check_same_thread": False}
)

# Instantiate SQLDatabase objects:
db_for_schema = SQLDatabase(engine=full_engine, sample_rows_in_table_info=5)
schema_info = db_for_schema.get_table_info()  # Detailed schema info
db = SQLDatabase(engine=ro_engine)

######################################
# 3. USER CONTROLS (Debug & Extra Columns)
######################################

# Toggle for displaying the generated SQL query for debugging.
show_sql_debug = st.sidebar.checkbox("Show SQL Query for debugging", value=False)

# Sidebar multiselect for extra columns that should always be included.
# (These options are taken from the uploaded file's columns.)
extra_columns = st.sidebar.multiselect(
    "Always display these columns (if applicable)", 
    options=df.columns.tolist(), 
    default=[]
)

######################################
# 4. INITIALIZE THE LLM & PROMPT TEMPLATE
######################################

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
    model=r"qwen2.5-coder-7b-instruct",
    temperature=0.0,
)

# Updated prompt template now includes extra_columns as part of the instructions.
prompt_template_text = '''
    You are a top tier data analyst that strives to understand the customer's needs and provide the most relevant data.   

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

# Prepare prompt data. We join the extra_columns list into a comma-separated string.
prompt_data = {
    "dialect": db.dialect,
    "table_info": schema_info,
    "input": "{input}",
    "extra_columns": ", ".join(extra_columns) if extra_columns else "None"
}

query_prompt_template = ChatPromptTemplate.from_template(prompt_template_text)

# Define a typed dict for the structured output.
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

######################################
# 6. STREAMLIT USER INTERFACE FOR QA
######################################

st.markdown("### Ask a question about your data (SQL retrieval only):")
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
    
    # # Optionally display intermediate debugging responses.
    # if show_sql_debug:
    #     st.subheader("Raw SQL JSON Result")
    #     st.code(sql_state["result"], language="json")
