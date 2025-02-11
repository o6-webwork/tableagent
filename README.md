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
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Create a Virtual Environment (optional but recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Required Packages
Either create a `requirements.txt` with the dependencies or install them individually:
```sh
pip install streamlit pandas sqlalchemy langchain streamlit-aggrid numpy
```
**Note:** You may need to install additional packages depending on your LLM server configuration.

### 4. Set Up Environment Variables
Set your API key and other credentials (if needed). For example:
```sh
export OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

### 1. Run the Streamlit App
```sh
streamlit run table_rag.py
```

### 2. Upload Your Data
- On the main page, use the file uploader to select a CSV, XLSX, or JSON file.
- The data is ingested, and a SQLite database is built.
- **Note:** If you upload a new file, the app automatically clears previously cached data (including the DataFrame, table name, sample data, and data dictionary).

### 3. Manage the Data Dictionary
- If enabled, the app will generate a data dictionary (using the table schema from the SQL database and sample data).
- You can edit the data dictionary using the provided interface and then save it.
- The data dictionary is merged with the table schema to form a combined JSON schema for the LLM prompt.

### 4. Ask Questions
- Enter a natural language question in the text input.
- The LLM generates a SQL query based on the combined schema and the question.
- The query is executed against the SQLite database, and the results are displayed using `st-aggrid`.
- If enabled, a query explanation is also generated and displayed in an expander.
- A debug toggle allows you to view the full SQL generation prompt.

### 5. Download Results
- Download buttons are available to export the query results in CSV, Excel, or JSON formats.

## Configuration

### LLM Settings
Adjust the parameters of the `ChatOpenAI` instance in the code (`base_url`, `api_key`, `model`, `temperature`) to match your local LLM server.

### SQL Database
- The database is created dynamically with a unique session ID.
- The table name is derived from the uploaded file name.
- The combined schema JSON is built by merging:
  - The table schema (from SQLite PRAGMA),
  - The optional data dictionary,
  - Sample data extracted from the table.

### Debug Options
- Toggle **Show SQL Query** and **Generate Query Explanation** on the sidebar.
- **Show SQL Generation Prompt (Debug Options)** displays the prompt text sent to the LLM.

## Future Enhancements

### Multi-Table Support
- The combined schema JSON is structured with a `tables` array to allow for future support of multiple tables and relationships.

### Advanced Metadata Management
- Although SQLite does not natively support column comments, you could integrate a custom metadata table for CRUD operations on column descriptions.

### Enhanced Query Validation
- Further post-processing and SQL parsing could be implemented to enforce constraints (e.g., ensuring only `SELECT` statements and case-insensitive comparisons).

## Contributing

Contributions and improvements are welcome! Please submit issues or pull requests to help enhance functionality, fix bugs, or improve documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

Special thanks to the developers of Streamlit, LangChain, and the HuggingFace community for providing the robust tools and libraries that power TableAgent.
