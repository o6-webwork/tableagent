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

# For semantic retrieval (using a local embedding model)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

######################################
# 1. GENERALIZED FILE INPUT & DATA LOAD
######################################

st.title("TableRAG: SQL + Semantic Search for Tabular Data")
st.markdown(
    """
    Upload a CSV, XLSX, or JSON file. The app will ingest the data, build a database, and allow you to ask questions.
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
# One for schema introspection using full permissions…
db_for_schema = SQLDatabase(engine=full_engine)
schema_info = db_for_schema.get_table_info()  # This returns detailed schema info

# …and one for executing queries in read-only mode.
db = SQLDatabase(engine=ro_engine)

######################################
# 3. USER CONTROLS (Debug & Weighting)
######################################

# Toggle for displaying the generated SQL query for debugging.
show_sql_debug = st.sidebar.checkbox("Show SQL Query for debugging", value=False)

# Slider to adjust the relative weighting of SQL vs. semantic retrieval (0 means all semantic, 1 means all SQL)
sql_weight = st.sidebar.slider("SQL Weight (0: semantic, 1: SQL)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Number of top results to retrieve
top_n = st.sidebar.number_input("Top N (SQL and semantic retrieval)", min_value=1, max_value=100, value=10, step=1)

######################################
# 4. INITIALIZE THE LLM & PROMPT TEMPLATES
######################################

# Initialize your local LLM (chat model) instance.
# Be sure to replace the API key and model path as needed.
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
    model=r"qwen2.5-7b-instruct",
)

# Define the prompt template for generating SQL queries.
query_prompt_template = ChatPromptTemplate.from_template(
    '''
    Given an input question, create a syntactically correct {dialect} query to help find the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

    Never query for all columns from a specific table; only select a few relevant columns given the question.

    The query should be formatted as JSON with a single field "query":

    {{
      "query": "SELECT COUNT(*) FROM Database;"
    }}

    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

    Only use the following tables:
    {table_info}

    Question: {input}
    '''
)

# Define a typed dict for the structured output from the LLM.
class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

######################################
# 5. SQL CHAIN FUNCTIONS
######################################

def generate_sql_query(question: str, top_k: int) -> dict:
    """Generate a SQL query based on the user question and the full schema info."""
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": top_k,
        "table_info": schema_info,
        "input": question,
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    if show_sql_debug:
        st.write("DEBUG: SQL prompt:", prompt)
        st.write("DEBUG: Generated SQL:", result["query"])
    return {"query": result["query"]}

def execute_sql_query(state: dict) -> dict:
    """Execute the generated SQL query in a read-only manner."""
    query = state["query"]
    if not query.strip().lower().startswith("select"):
        return {"result": "Error: Only SELECT queries are permitted."}
    with ro_engine.connect() as conn:
        result = conn.execute(text(query))
        # Use the _mapping attribute to convert each row to a dict (SQLAlchemy 1.4+)
        rows = [dict(row._mapping) for row in result]
    return {"result": json.dumps(rows, indent=2)}

def generate_sql_answer(state: dict) -> dict:
    """Use the SQL result and query to generate a final answer."""
    prompt = (
        "Given the following information:\n\n"
        f"User Question: {state['question']}\n\n"
        f"SQL Query: {state['query']}\n\n"
        f"SQL Result: {state['result']}\n\n"
        "Please generate a clear and concise answer to the user question."
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

######################################
# 6. SEMANTIC RETRIEVAL (RAG)
######################################

@st.cache_data(show_spinner=False)
def create_documents(dataframe: pd.DataFrame):
    """Convert each row of the DataFrame into a text document for semantic search."""
    docs = []
    for _, row in dataframe.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in dataframe.columns])
        docs.append(Document(page_content=text))
    return docs

documents = create_documents(df)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)

def semantic_retrieval_answer(question: str, top_k: int) -> str:
    """Retrieve context documents semantically and generate an answer."""
    retrieved_docs = vector_store.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = (
        "You are given the following context documents:\n\n"
        f"{context}\n\n"
        "Based on the context above, answer the following question:\n\n"
        f"{question}"
    )
    response = llm.invoke(prompt)
    return response.content

######################################
# 7. COMBINE THE ANSWERS (BEST-OF-BOTH-WORLDS)
######################################

def best_of_worlds(sql_ans: str, semantic_ans: str, question: str, sql_weight: float) -> str:
    """
    Combine the SQL-based and semantic answers.
    
    If the question appears to be asking for a count (e.g. "how many", "count", "total", etc.),
    the SQL answer is likely more reliable and is prioritized.
    
    Otherwise, a weighted combination is requested.
    """
    lower_q = question.lower()
    if any(kw in lower_q for kw in ["how many", "count", "number of", "total"]):
        prompt = (
            "You are given a user question and two answers: one from SQL-based retrieval and one from semantic retrieval.\n"
            "Since the question appears to be asking for a numerical count, the SQL-based answer is likely more reliable.\n"
            "SQL-based answer: {sql_ans}\n"
            "Semantic retrieval answer: {semantic_ans}\n"
            "Please provide a final answer that prioritizes the SQL-based result."
        ).format(sql_ans=sql_ans, semantic_ans=semantic_ans)
        response = llm.invoke(prompt)
        return response.content
    else:
        prompt = (
            "You are given a user question and two answers: one from SQL-based retrieval and one from semantic retrieval.\n"
            "The SQL answer should be weighted at {sql_weight} and the semantic answer at {semantic_weight}.\n"
            "SQL-based answer: {sql_ans}\n"
            "Semantic retrieval answer: {semantic_ans}\n"
            "Question: {question}\n"
            "Please combine these answers into a final answer that takes these weights into account."
        ).format(
            sql_weight=sql_weight,
            semantic_weight=1 - sql_weight,
            sql_ans=sql_ans,
            semantic_ans=semantic_ans,
            question=question
        )
        response = llm.invoke(prompt)
        return response.content

######################################
# 8. STREAMLIT USER INTERFACE FOR QA
######################################

st.markdown(
    """
    ### Ask a question about your data:
    """
)
user_question = st.text_input("Your question:")

if st.button("Submit") and user_question:
    st.write("Processing your question...")
    
    # --- SQL Retrieval Chain ---
    sql_state = {"question": user_question}
    sql_state.update(generate_sql_query(user_question, top_n))
    sql_state.update(execute_sql_query(sql_state))
    sql_state.update(generate_sql_answer(sql_state))
    sql_based_answer = sql_state["answer"]
    
    # --- Semantic Retrieval Chain ---
    semantic_based_answer = semantic_retrieval_answer(user_question, top_n)
    
    # --- Combine Answers with Weighting ---
    final_answer = best_of_worlds(sql_based_answer, semantic_based_answer, user_question, sql_weight)
    
    # Display outputs
    st.subheader("SQL-based Answer")
    st.code(sql_based_answer, language="text")
    
    st.subheader("Semantic Retrieval Answer")
    st.code(semantic_based_answer, language="text")
    
    st.subheader("Final Combined Answer")
    st.success(final_answer)
