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

st.title("TableRAG: Hybrid FAISS ⟷ SQL Retrieval for Tabular Data")
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
schema_info = db_for_schema.get_table_info()  # Detailed schema info

# …and one for executing queries in read-only mode.
db = SQLDatabase(engine=ro_engine)

######################################
# 3. USER CONTROLS (Debug & Intermediate Toggle)
######################################

# Toggle for displaying the generated SQL query for debugging.
show_sql_debug = st.sidebar.checkbox("Show SQL Query for debugging", value=False)

# Toggle for showing intermediate SQL and Semantic responses.
show_intermediate = st.sidebar.checkbox("Show intermediate SQL & Semantic responses", value=False)

# Number of top results to retrieve.
top_n = st.sidebar.number_input("Top N (SQL and semantic retrieval)", min_value=1, max_value=100, value=10, step=1)

######################################
# 4. INITIALIZE THE LLM & PROMPT TEMPLATES
######################################

# Initialize your local LLM (chat model) instance.
# Adjust the base_url, API key, and model parameters as needed.
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
    """Execute the generated SQL query in a read-only manner and return row count."""
    query = state["query"]
    if not query.strip().lower().startswith("select"):
        return {"result": "Error: Only SELECT queries are permitted.", "row_count": 0}
    with ro_engine.connect() as conn:
        result = conn.execute(text(query))
        # Convert rows using _mapping (SQLAlchemy 1.4+)
        rows = [dict(row._mapping) for row in result]
        row_count = len(rows)
    return {"result": json.dumps(rows, indent=2), "row_count": row_count}

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
    """Convert each row of the DataFrame into a text document with metadata for semantic search."""
    docs = []
    for _, row in dataframe.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in dataframe.columns])
        metadata = row.to_dict()  # include the entire row as metadata
        docs.append(Document(page_content=text, metadata=metadata))
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
# 7. QUERY CLASSIFICATION
######################################

def classify_query(query: str) -> str:
    """
    Use the LLM to classify the query into one of three categories:
      - "SQL": Structured query (e.g., filtering, aggregations).
      - "Semantic": Purely unstructured semantic search.
      - "Hybrid": Contains both structured and unstructured elements.
    """
    prompt = f"""
    Determine the retrieval nature of the following query. Respond with one word: "SQL", "Semantic", or "Hybrid".
    Query: "{query}"
    """
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()
    if "sql" in classification and "semantic" not in classification:
        return "SQL"
    elif "semantic" in classification and "sql" not in classification:
        return "Semantic"
    else:
        return "Hybrid"

######################################
# 8. COMBINE THE ANSWERS (HEURISTIC FUSION)
######################################

def combine_answers(sql_query: str, sql_ans: str, semantic_ans: str, question: str, row_count: int) -> str:
    """
    Combine the SQL-based and semantic answers using heuristic logic.

    Heuristics:
      1. If the SQL answer is unsuccessful (empty, error, or no rows found), use the semantic answer.
      2. If the SQL query contains a LIKE operator (approximate matching),
         instruct the LLM to weigh the semantic answer more heavily.
      3. Otherwise, assume the SQL answer is accurate and use it as the primary factual source.
    """
    normalized_sql_query = sql_query.lower()
    normalized_sql_ans = sql_ans.strip().lower()
    
    # Heuristic 1: No useful SQL results.
    if row_count == 0 or (not normalized_sql_ans) or ("error" in normalized_sql_ans) or (normalized_sql_ans in {"[]", "null"}):
        prompt = (
            "The SQL query did not return any useful results. "
            "The semantic search output is as follows:\n\n"
            f"Semantic Retrieval Answer: {semantic_ans}\n\n"
            "Based on this, provide a final answer to the question:\n\n"
            f"Question: {question}"
        )
        response = llm.invoke(prompt)
        return response.content

    # Heuristic 2: SQL query uses a LIKE operator.
    if "like" in normalized_sql_query:
        prompt = (
            "The SQL query used an approximate matching operator (LIKE), indicating that it "
            "attempted to approximate a semantic query. The SQL answer is as follows:\n\n"
            f"SQL-based Answer: {sql_ans}\n\n"
            "The semantic retrieval provided the following result:\n\n"
            f"Semantic Retrieval Answer: {semantic_ans}\n\n"
            "Given these two answers, provide a final answer that heavily weighs the semantic search output for context "
            "while still incorporating the factual details from the SQL answer where appropriate.\n\n"
            f"Question: {question}"
        )
        response = llm.invoke(prompt)
        return response.content

    # Heuristic 3: SQL answer is clear.
    prompt = (
        "The SQL query was successful and returned a clear answer. The outputs are as follows:\n\n"
        f"SQL-based Answer: {sql_ans}\n\n"
        f"Semantic Retrieval Answer: {semantic_ans}\n\n"
        "Provide a final answer that uses the SQL-based answer as the primary factual source, "
        "enriching it with any additional context from the semantic search if needed.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    return response.content

######################################
# 9. STREAMLIT USER INTERFACE FOR QA
######################################

st.markdown("### Ask a question about your data:")
user_question = st.text_input("Your question:")

if st.button("Submit") and user_question:
    st.write("Processing your question...")
    
    # Classify the query to determine which retrieval pipeline(s) to run.
    query_type = classify_query(user_question)
    st.write(f"**Query Classification:** {query_type.capitalize()}")
    
    if query_type == "sql":
        # Run only the SQL retrieval chain.
        sql_state = {"question": user_question}
        sql_state.update(generate_sql_query(user_question, top_n))
        sql_exec = execute_sql_query(sql_state)
        sql_state.update(sql_exec)
        sql_state.update(generate_sql_answer(sql_state))
        final_answer = sql_state["answer"]
        semantic_based_answer = None
    elif query_type == "semantic":
        # Run only the semantic retrieval chain.
        final_answer = semantic_retrieval_answer(user_question, top_n)
        sql_state = None
        semantic_based_answer = final_answer
    else:  # Hybrid
        # Run both SQL and semantic retrieval chains.
        sql_state = {"question": user_question}
        sql_state.update(generate_sql_query(user_question, top_n))
        sql_exec = execute_sql_query(sql_state)
        sql_state.update(sql_exec)
        sql_state.update(generate_sql_answer(sql_state))
        sql_based_answer = sql_state["answer"]
        sql_row_count = sql_exec.get("row_count", 0)
        semantic_based_answer = semantic_retrieval_answer(user_question, top_n)
        final_answer = combine_answers(
            sql_query=sql_state["query"],
            sql_ans=sql_state["answer"],
            semantic_ans=semantic_based_answer,
            question=user_question,
            row_count=sql_row_count
        )
    
    # Display the final answer.
    st.subheader("Final Answer")
    st.success(final_answer)
    
    # Optionally display intermediate results if toggle is enabled.
    if show_intermediate:
        if sql_state is not None:
            st.subheader("Intermediate SQL-based Answer")
            st.code(sql_state["answer"], language="text")
            st.subheader("SQL Query")
            st.code(sql_state["query"], language="sql")
        if semantic_based_answer is not None:
            st.subheader("Intermediate Semantic Retrieval Answer")
            st.code(semantic_based_answer, language="text")
