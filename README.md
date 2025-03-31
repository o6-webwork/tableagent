# TableAgent

TableAgent is a Streamlit-based application that ingests CSV, XLSX, or JSON files to build a SQLite database and leverages a local language model (LLM) to generate SQL queries from natural language questions. The generated queries are executed against the database, and the results—as well as a human-readable explanation of the query—are presented to the user. In addition, the app automatically constructs a combined JSON schema for the table (with details from the SQL metadata, sample data, and an optional data dictionary) that helps guide the LLM to generate accurate, case-insensitive SQL queries. The design is extensible to support multiple tables in the future.

## Features

### Generalized Data Ingestion
- Upload CSV, XLSX, or JSON files.
- The data is ingested and stored in a dynamically named SQLite database.

### Dynamic Schema Generation
- Retrieves table metadata using SQLite’s PRAGMA commands.
- Merges the table schema with a user-editable data dictionary and sample data.
- Constructs a combined JSON schema with fields for each column:
  - `column_name`
  - `data_type`
  - `column_description`
  - `sample_data` (up to 3 randomly sampled example values)
  - `has_null` (a boolean indicating whether null values are present)
- The JSON is structured with a `tables` array to facilitate multi-table support in the future.

### LLM-Based SQL Query Generation and Explanation
- Uses a local LLM (via LangChain) to convert natural language questions into SQL queries.
- Enforces constraints in the prompt so that the generated query:
  - Only uses `SELECT` statements.
  - References only column names from the combined JSON schema.
  - Performs case-insensitive text comparisons.
- Provides a query explanation in plain language.

### Debug and Data Dictionary Editing
- A debug toggle displays the SQL generation prompt.
- A built-in CRUD interface allows you to view and edit the generated data dictionary (column metadata).

### Result Display and Download
- Executes generated SQL queries on a read-only SQLite connection.
- Displays results using `st-aggrid`.
- Provides options to download query results in CSV, Excel, or JSON format.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- SQLAlchemy
- LangChain (and community extensions)
- HuggingFace Transformers (or any compatible local LLM server)
- `st-aggrid`
- Other standard libraries (random, numpy, etc.)

Additional dependencies might be required by your local LLM server.

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/o6-webwork/tableagent.git
cd tableagent
```

### 2. Run with Docker (Recommended)
To avoid dependency conflicts, you can run TableAgent inside a Docker container.

#### **Building the Docker Image**
```sh
docker build -t tableagent .
```

#### **Running the Container**
```sh
docker run --rm -p 8501:8501 tableagent
```
- `--rm` removes the container automatically after it stops, keeping things clean.
- `-p 8501:8501` maps the container's port to the host, making the Streamlit app accessible at `http://localhost:8501`.

### **Accessing the App on a Local Network (LAN)**
If you want to access the app from another device on the same network:
1. Find your host machine's local IP address:
   - **Mac/Linux:** `ip a | grep inet`
   - **Windows:** `ipconfig`
   - Look for an IP address like `192.168.x.x` or `10.x.x.x`.
2. On another device connected to the same network, open a browser and visit:
   ```
   http://<HOST_IP>:8501
   ```
   Example:
   ```
   http://192.168.1.100:8501
   ```

#### **Stopping the Container**
Press `Ctrl+C` in the terminal running the container, or stop it using:
```sh
docker ps  # Find the container ID
docker stop <container_id>
```

### 3. Manual Installation (Without Docker)
If you prefer running locally, follow these steps:

#### **Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **Install Dependencies**
```sh
pip install -r requirements.txt
```

#### **Run the App**
```sh
streamlit run table_rag.py
```
