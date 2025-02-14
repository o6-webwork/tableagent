import streamlit as st
import pandas as pd
import os
import json
import uuid
import time
import io  # For Excel download conversion
import random  # For random sampling
import numpy as np  # For handling numpy.bool_
import re
import requests
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
# Helper: Sanitise names (filenames, columns)
######################################
def sanitize_name(name: str) -> str:
    # Replace any non-word character with underscore
    return re.sub(r'\W+', '_', name)

######################################
# SESSION SETUP: Unique DB per User Session
######################################

TEMP_DB_DIR = "temp_db"
if not os.path.exists(TEMP_DB_DIR):
    os.makedirs(TEMP_DB_DIR)

def cleanup_old_session_dbs(folder, expiration_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            mtime = os.path.getmtime(filepath)
            if now - mtime > expiration_seconds:
                os.remove(filepath)

cleanup_old_session_dbs(TEMP_DB_DIR)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]
db_file = os.path.join(TEMP_DB_DIR, f"{session_id}.db")
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

######################################
# Sidebar Options
######################################
@st.dialog("Error!")
def error_popup(e):
    st.write(f"{e}")
    refresh_button = st.button("Got it!")
    if refresh_button:
        st.rerun()

st.sidebar.header("LLM Configuration")
openapiurl = st.sidebar.text_input("Base URL", "http://127.0.0.1:1234/v1")
openapitoken = st.sidebar.text_input("API Key", os.environ.get("OPENAI_API_KEY", "token-abc123"))
st.session_state.openapitoken = openapitoken
st.session_state.openaiapiurl = openapiurl 

load_models = st.sidebar.button("Load models" if 'model_list' not in st.session_state else "Refresh models")
if load_models:
    url = st.session_state.openaiapiurl + "/models"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openapitoken}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Parse the JSON response and extract model IDs
            response_data = response.json()
            model_list = [model["id"] for model in response_data["data"]]
            st.session_state.model_list = model_list
        elif response.status_code == 401:
            raise Exception(f"Response status {response.status_code} Unauthorized, Check your API token")
        else:
            raise Exception(f"Response status {response.status_code}")
    except Exception as e:
        error_popup(e)

# Always show the model select box if the model list is available
if 'model_list' in st.session_state:
    model_list = st.session_state.model_list
    selected_model = st.sidebar.selectbox("Choose model", model_list)
    st.session_state.selected_model = selected_model

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

#Safe mode toggle in the sidebar (default True)
st.sidebar.header("Safe Mode Options")
safe_mode = st.sidebar.checkbox("Enable Safe Mode (read-only)", value=True)
st.session_state["safe_mode"] = safe_mode

st.sidebar.header("Explainability Options")
#show_sql_debug = st.sidebar.checkbox("Show SQL Query", value=False)
query_explainability_toggle = st.sidebar.checkbox("Generate Query Explanation", value=False)

st.sidebar.header("Data Dictionary")
data_dictionary_toggle = st.sidebar.checkbox("Generate and use Data Dictionary", value=False)

st.sidebar.header("Debug Options")
show_prompt_toggle = st.sidebar.checkbox("Show SQL Generation Prompt", value=False)

# Ensure a model has been selected before initializing ChatOpenAI
if 'selected_model' not in st.session_state:
    st.error("Please load and select a model.")
else:
    llm = ChatOpenAI(
        base_url=openapiurl,
        api_key=openapitoken,
        model=st.session_state.selected_model,
        temperature=temperature,
    )

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json"])
if uploaded_file is not None:
    # If this is a new file (different filename), clear cached keys
    if "uploaded_file_name" not in st.session_state or st.session_state["uploaded_file_name"] != uploaded_file.name:
        keys_to_clear = ["df", "table_name", "sample_data", "data_dictionary", "generated_query"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.session_state["uploaded_file_name"] = uploaded_file.name
    # Continue processing the file...
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
        # Sanitize column names
        df.columns = [sanitize_name(col) for col in df.columns]
        if df.columns.duplicated().any():
            st.error("Duplicate column names detected after processing. Please check the file.")
            st.stop()
        # Sanitize table name
        table_name = sanitize_name(os.path.splitext(uploaded_file.name)[0])
        st.session_state["table_name"] = table_name
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    if "df" in st.session_state:
        df = st.session_state["df"]
    else:
        st.info("Please upload a CSV, XLSX, or JSON file to proceed.")
        st.stop()

######################################
# 2. SET UP THE DATABASE ENGINES
######################################

if "full_engine" not in st.session_state:
    full_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    st.session_state["full_engine"] = full_engine
else:
    full_engine = st.session_state["full_engine"]

table_name = st.session_state.get("table_name", "training")

if "table_loaded" not in st.session_state:
    df.to_sql(table_name, full_engine, if_exists="replace", index=False)
    st.session_state["table_loaded"] = True

if "ro_engine" not in st.session_state:
    ro_engine = create_engine(
        f"sqlite:///file:{db_file}?mode=ro&uri=true",
        connect_args={"check_same_thread": False}
    )
    st.session_state["ro_engine"] = ro_engine
else:
    ro_engine = st.session_state["ro_engine"]

# Get table schema from SQL DB using PRAGMA
with ro_engine.connect() as conn:
    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
    table_schema = [dict(row._mapping) for row in result]

db_for_schema = SQLDatabase(engine=full_engine, sample_rows_in_table_info=5)
schema_info = db_for_schema.get_table_info()  # We still keep this for legacy purposes if needed.
db = SQLDatabase(engine=ro_engine)

######################################
# Helper Function: Generate Combined Schema JSON for multiple tables
######################################
def generate_combined_schema(table_schema, data_dictionary, sample_data, df, table_name):
    columns = []
    for col_info in table_schema:
        col_name = col_info["name"]
        data_type = col_info["type"]
        col_description = data_dictionary.get(col_name, "") if data_dictionary else ""
        sample_values = []
        if sample_data:
            for row in sample_data:
                if col_name in row and row[col_name] is not None:
                    sample_values.append(str(row[col_name]))
            sample_values = list(dict.fromkeys(sample_values))
            if len(sample_values) > 3:
                sample_values = random.sample(sample_values, 3)
        has_null = bool(df[col_name].isnull().any())
        columns.append({
            "column_name": col_name,
            "data_type": data_type,
            "column_description": col_description,
            "sample_data": sample_values,
            "has_null": has_null
        })
    combined = {
        "tables": [
            {
                "table_name": table_name,
                "columns": columns
            }
        ]
    }
    return json.dumps(combined, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))

######################################
# 3. DATA DICTIONARY: GENERATE AND EDIT (with Sample Data refresh)
######################################
if data_dictionary_toggle:
    if st.button("Refresh Sample Data", key="refresh_sample"):
        with ro_engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 5"))
            sample_rows = [dict(row._mapping) for row in result]
        st.session_state["sample_data"] = sample_rows
    if "sample_data" not in st.session_state:
        with ro_engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 5"))
            sample_rows = [dict(row._mapping) for row in result]
        st.session_state["sample_data"] = sample_rows

    st.markdown("### Sample Data (5 rows)")
    sample_df = pd.DataFrame(st.session_state["sample_data"])
    gb_sample = GridOptionsBuilder.from_dataframe(sample_df)
    gb_sample.configure_default_column(resizable=True, wrapText=True, autoHeight=True, minWidth=125)
    gridOptions_sample = gb_sample.build()
    # gridOptions_sample["onGridReady"] = """
    #     function(params) {
    #     const allColumnIds = params.columnApi.getAllColumns().map(col => col.colId);
    #     params.columnApi.autoSizeColumns(allColumnIds);
    #     }
    #     """
    AgGrid(
        sample_df,
        gridOptions=gridOptions_sample,
        key="sample_data_aggrid",
        height=300,
        fit_columns_on_grid_load=True
    )
    
    if "data_dictionary" not in st.session_state:
        with st.spinner("Generating data dictionary..."):
            data_dict_prompt = '''
You are an expert data analyst. Given the following schema information and sample data, generate a short description for what each column represents.

Schema info:
{schema_info}

Sample data (first few rows):
{sample_data}

Output the result as JSON with a single field "explanations", which is an object where keys are column names and values are the descriptions.
            '''
            data_dict_prompt_template = ChatPromptTemplate.from_template(data_dict_prompt)

            class DataDictionaryOutput(TypedDict):
                explanations: Annotated[dict, ..., "Mapping of column names to their descriptions."]

            def generate_data_dictionary() -> dict:
                sample_data = json.dumps(st.session_state["sample_data"], indent=2)
                prompt = data_dict_prompt_template.invoke({
                    "schema_info": json.dumps(table_schema, indent=2),
                    "sample_data": sample_data
                })
                structured_llm = llm.with_structured_output(DataDictionaryOutput)
                result = structured_llm.invoke(prompt)
                return result["explanations"]

            st.session_state["data_dictionary"] = generate_data_dictionary()

    st.markdown("### Data Dictionary (Edit as needed)")
    data_dict_df = pd.DataFrame(
        list(st.session_state["data_dictionary"].items()),
        columns=["Column", "Description"]
    )
    edited_data_dict_df = st.data_editor(
        data_dict_df,
        key="data_dict_editor",
        use_container_width=True,
        hide_index=True
    )
    if st.button("Save Data Dictionary"):
        st.session_state["data_dictionary"] = dict(zip(edited_data_dict_df["Column"], edited_data_dict_df["Description"]))
        st.success("Data Dictionary saved!")

######################################
# 4. INITIALIZE PROMPT TEMPLATES FOR SQL & QUERY EXPLANATION
######################################
prompt_template_text = '''
You are a top tier data analysis algorithm that strives to understand the customer's needs and provide the most relevant data.

Given the customer's input question, generate a syntactically correct {dialect} SQL query that retrieves only the necessary columns to answer the question. Do NOT include a LIMIT clause in your query.

The output must be formatted as JSON with a single field "query". For example:

{{
  "query": "SELECT COUNT(*) FROM Database;"
}}

Mistakes are very costly, so use the following combined schema information (in JSON format) to ensure you reference only columns that exist. The JSON is structured as follows:
- It has a "tables" field which is an array.
- Each element represents a table with:
   - "table_name": the table's name,
   - "columns": an array of objects, each with:
       - "column_name": the name of the column,
       - "data_type": the column's data type,
       - "column_description": a description of what the column represents; pay careful attention to this description when selecting columns,
       - "sample_data": a list of up to 3 example values,
       - "has_null": a boolean indicating if null values are present.

<combined_schema_info>
{combined_schema_info}
</combined_schema_info>
    
Question: {input}

Ensure that any text comparisons in your query are case-insensitive. For example, use LOWER(column) = LOWER('value') or apply a COLLATE NOCASE clause when filtering.

Show all columns whenever possible unless you are sure that the user is asking for very specific information, or the query involves an aggregate function (e.g. COUNT, MIN, MAX, AVG, SUM).
'''
query_prompt_template = ChatPromptTemplate.from_template(prompt_template_text)

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

query_explain_prompt_template_text = '''
You are an expert SQL analyst. Given the following SQL query and the original question, provide a clear and concise layman explanation of what the query does and how it answers the question. Include details on any aggregation, filtering, or ordering that the query performs. 

SQL Query:
{query}

Original Question:
{question}

Output the result as JSON with a single field "explanation".
'''
query_explain_prompt_template = ChatPromptTemplate.from_template(query_explain_prompt_template_text)

class QueryExplanationOutput(TypedDict):
    explanation: Annotated[str, ..., "Explanation of what the SQL query does."]

######################################
# 5. SQL CHAIN FUNCTIONS
######################################
def generate_sql_query(question: str) -> dict:
    combined_schema_info = generate_combined_schema(
        table_schema,
        st.session_state.get("data_dictionary", {}),
        st.session_state.get("sample_data", []),
        df,
        st.session_state.get("table_name", "training")
    )
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "combined_schema_info": combined_schema_info,
        "input": question,
        #"extra_columns": ", ".join(extra_columns) if extra_columns else "None"
    })
    if show_prompt_toggle:
         st.markdown("### Debug: SQL Generation Prompt")
         st.code(prompt, language="text")
    structured_llm = llm.with_structured_output(QueryOutput)
    with st.spinner("Generating SQL query..."):
        result = structured_llm.invoke(prompt)
    query_generated = result["query"]
    st.session_state["generated_query"] = query_generated
    st.subheader("Generated SQL Query")
    st.code(query_generated, language="sql")
    return {"query": query_generated}

def generate_query_explanation(query: str, question: str) -> str:
    prompt = query_explain_prompt_template.invoke({
        "query": query,
        "question": question
    })
    structured_llm = llm.with_structured_output(QueryExplanationOutput)
    with st.spinner("Generating SQL explanation..."):
        result = structured_llm.invoke(prompt)
    return result["explanation"]

def execute_sql_query(state: dict) -> dict:
    query = state["query"]
    safe_mode = st.session_state.get("safe_mode", True)
    # Use the appropriate engine based on safe mode toggle.
    engine_to_use = ro_engine if safe_mode else full_engine
    try:
        with engine_to_use.connect() as conn:
            result = conn.execute(text(query))
            rows = [dict(row._mapping) for row in result]
            row_count = len(rows)
        return {"result": json.dumps(rows, indent=2), "row_count": row_count}
    except Exception as e:
        error_message = str(e)
        # Gracefully handle the read-only error if safe mode is enabled.
        if safe_mode and "attempt to write a readonly database" in error_message:
            st.error("This query is prohibited under safe mode to prevent data modification.")
            return {"result": json.dumps([]), "row_count": 0}
        else:
            st.error(f"Error executing query: {error_message}")
            # Attempt error correction by feeding error back to LLM to regenerate the query.
            corrected_query = regenerate_sql_query(state["question"], error_message, state["query"])
            if corrected_query:
                st.info("Regenerated corrected query:")
                st.code(corrected_query, language="sql")
                state["query"] = corrected_query
                # Update the generated query in session state so that query explanation uses it.
                st.session_state["generated_query"] = corrected_query
                try:
                    with engine_to_use.connect() as conn:
                        result = conn.execute(text(corrected_query))
                        rows = [dict(row._mapping) for row in result]
                        row_count = len(rows)
                    return {"result": json.dumps(rows, indent=2), "row_count": row_count}
                except Exception as e2:
                    st.error(f"Error executing corrected query: {str(e2)}")
                    return {"result": json.dumps([]), "row_count": 0}
            else:
                return {"result": json.dumps([]), "row_count": 0}

def regenerate_sql_query(question: str, error_message: str, original_query: str) -> str:
    # Rebuild combined schema info for error correction
    combined_schema_info = generate_combined_schema(
        table_schema,
        st.session_state.get("data_dictionary", {}),
        st.session_state.get("sample_data", []),
        df,
        st.session_state.get("table_name", "training")
    )
    correction_prompt = f'''
The following SQL query produced an error:
SQL Query: {original_query}
Error: {error_message}

Using the combined schema information below, generate a corrected SQL query that references only valid columns. The query must be a SELECT query.

Combined Schema Information:
{combined_schema_info}

Original Question: {question}

Please provide only the corrected SQL query.
    '''
    structured_llm = llm.with_structured_output(QueryOutput)
    with st.spinner("Regenerating corrected SQL query..."):
        try:
            result = structured_llm.invoke(correction_prompt)
            corrected_query = result["query"]
            return corrected_query
        except Exception as ex:
            st.error(f"Failed to regenerate query: {str(ex)}")
            return ""

def is_modification_query(query: str) -> bool:
    """Return True if the query starts with INSERT, UPDATE, or DELETE."""
    return bool(re.match(r'^\s*(INSERT|UPDATE|DELETE)', query, re.IGNORECASE))

def add_returning_clause(query: str) -> str:
    """If the query is a modification query and does not include a RETURNING clause, add one."""
    if is_modification_query(query):
        if "returning" not in query.lower():
            # Remove any trailing semicolon and add " RETURNING *;"
            query = query.rstrip().rstrip(';')
            query += " RETURNING *;"
    return query

def preview_modification_query(query: str) -> list:
    """
    Execute the query in a transaction and roll it back so that we can preview 
    the modified rows without actually changing the database.
    """
    query_with_returning = add_returning_clause(query)
    with full_engine.connect() as conn:
        trans = conn.begin()  # begin a transaction
        try:
            result = conn.execute(text(query_with_returning))
            preview_rows = [dict(row._mapping) for row in result]
        except Exception as e:
            preview_rows = [{"error": str(e)}]
        trans.rollback()  # rollback any changes
    return preview_rows

def execute_modification_query(query: str) -> dict:
    query_with_returning = add_returning_clause(query)
    rows = []
    row_count = None
    with full_engine.connect() as conn:
        trans = conn.begin()
        try:
            result = conn.execute(text(query_with_returning))
            try:
                rows = [dict(row._mapping) for row in result]
            except Exception:
                rows = []
            row_count = result.rowcount
            trans.commit()
        except Exception as e:
            trans.rollback()
            # If the error is related to the RETURNING clause, try without it
            error_str = str(e)
            st.warning("Execution with RETURNING clause failed: " + error_str)
            st.warning("Trying to execute query without RETURNING clause...")
            query_without_returning = re.sub(r"(?i)\s+RETURNING\s+.*", "", query_with_returning).rstrip(";") + ";"
            with full_engine.connect() as conn2:
                trans2 = conn2.begin()
                try:
                    result2 = conn2.execute(text(query_without_returning))
                    row_count = result2.rowcount
                    trans2.commit()
                    rows = []  # Without RETURNING, we won’t have row details.
                except Exception as e2:
                    trans2.rollback()
                    st.error(f"Error executing modification query without RETURNING clause: {str(e2)}")
                    return {"rows": [], "row_count": 0}
    # Refresh the read-only engine so that subsequent queries see the changes.
    st.session_state["ro_engine"].dispose()
    st.session_state["ro_engine"] = create_engine(
        f"sqlite:///file:{db_file}?mode=ro&uri=true",
        connect_args={"check_same_thread": False}
    )
    return {"rows": rows, "row_count": row_count}

######################################
# 6. DOWNLOAD RESULTS FUNCTION
######################################
def download_results(df_result: pd.DataFrame):
    csv_bytes = df_result.to_csv(index=False).encode('utf-8')
    json_str = df_result.to_json(orient='records', indent=2)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Results')
    xlsx_bytes = buffer.getvalue()
    
    st.markdown("#### Download Results")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Excel",
        data=xlsx_bytes,
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="results.json",
        mime="application/json"
    )

######################################
# 7. STREAMLIT USER INTERFACE FOR QA
######################################
if "pending_mod_query" not in st.session_state:
    st.session_state["pending_mod_query"] = None
if "modification_confirmed" not in st.session_state:
    st.session_state["modification_confirmed"] = False

# --- Main Submit block ---
st.markdown("### Ask a question about your data:")
user_question = st.text_input("Your question:")

# When user clicks Submit, generate the query and store it
if st.button("Submit") and user_question:
    st.write("Processing your question...")
    sql_state = {"question": user_question}
    sql_state.update(generate_sql_query(user_question))
    query_generated = sql_state["query"]
    st.session_state["generated_query"] = query_generated

    # If it's a modification query (and safe mode is off), store it as pending
    if is_modification_query(query_generated) and not st.session_state.get("safe_mode", True):
        st.session_state["pending_mod_query"] = query_generated
        st.session_state["modification_confirmed"] = False  # Reset confirmation flag if needed

# --- Outside the Submit block, check if there's a pending modification query ---
if st.session_state.get("pending_mod_query"):
    query_generated = st.session_state["pending_mod_query"]
    # Only handle modification queries if safe mode is off
    if not st.session_state.get("safe_mode", True):
        st.warning("This query will modify your database.")

        st.markdown("#### Preview of Modified Rows")
        preview_rows = preview_modification_query(query_generated)
        if preview_rows and "error" not in preview_rows[0]:
            preview_rows = pd.DataFrame(preview_rows)
            gb = GridOptionsBuilder.from_dataframe(preview_rows)
            gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, minWidth = 125)
            gridOptions = gb.build()
            AgGrid(
                preview_rows, 
                gridOptions=gridOptions, 
                height=300, 
                fit_columns_on_grid_load=True, 
                key="aggrid_results", 
            )
        else:
            st.write("Could not retrieve preview:", preview_rows)

        # Ask for confirmation
        if not st.session_state.get("modification_confirmed"):
            if st.button("Confirm Execution of Modification Query", key="confirm_mod_query"):
                st.session_state["modification_confirmed"] = True
        else:
            st.write("Modification query confirmed.")

        # If confirmed, execute the modification query
        if st.session_state.get("modification_confirmed"):
            mod_result = execute_modification_query(query_generated)
            st.subheader("Modification Query Executed")
            st.success(f"{mod_result['row_count']} rows modified.")
            #if mod_result['rows']:
                #st.dataframe(pd.DataFrame(mod_result['rows']))
            # Clear the pending query once executed
            st.session_state["pending_mod_query"] = None
            st.session_state["modification_confirmed"] = False

# --- Else, handle non-modification queries as before ---
if st.session_state.get("generated_query") and (not is_modification_query(st.session_state["generated_query"]) or st.session_state.get("safe_mode", True)):
    # For non-modification queries (or when safe mode is on), execute as before.
    sql_state = {"query": st.session_state["generated_query"], "question": user_question}
    sql_state.update(execute_sql_query(sql_state))
    try:
        df_result = pd.read_json(io.StringIO(sql_state["result"]))
    except Exception as e:
        st.error(f"Error parsing SQL results: {e}")
        df_result = pd.DataFrame()

    st.subheader("Results")
    if not df_result.empty:
        gb = GridOptionsBuilder.from_dataframe(df_result)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, minWidth=125)
        gridOptions = gb.build()
        AgGrid(
            df_result, 
            gridOptions=gridOptions, 
            height=500, 
            fit_columns_on_grid_load=True, 
            key="aggrid_results", 
            update_mode=GridUpdateMode.NO_UPDATE
        )
    else:
        st.text("No results returned.")

    if query_explainability_toggle:
        with st.expander("Query Explanation", expanded=True):
            explanation = generate_query_explanation(st.session_state["generated_query"], user_question)
            st.markdown(explanation)

    download_results(df_result)