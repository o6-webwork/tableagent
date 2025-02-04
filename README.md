# TableRAG

This project implements a dual-retrieval chatbot using Streamlit. The chatbot accepts CSV, XLSX, or JSON files as data sources, builds a SQLite database, and uses two retrieval methods to answer user queries:

- **SQL-based retrieval:** Uses a language model (LLM) to generate SQL queries from natural language questions, executes them on a read-only database, and generates answers based on structured data.
- **Semantic retrieval (RAG):** Uses local embeddings (via HuggingFace and FAISS) to perform a similarity search on the data and generate context-based answers.
- **Combined Answering:** Depending on the question type (e.g., counting queries), the system prioritizes one method over the other. A slider allows you to adjust the weighting between the SQL and semantic answers.

A debug toggle is also available to display the generated SQL query for troubleshooting purposes.

## Features

- **Generalized Data Ingestion:** Accepts CSV, XLSX, or JSON files.
- **Dual Retrieval Approach:**
  - **SQL Retrieval:** Generates and executes SQL queries based on the data schema.
  - **Semantic Retrieval:** Uses embeddings to retrieve context and answer more open-ended questions.
- **Answer Combination Logic:** Dynamically prioritizes the SQL result for count/numeric questions or blends both answers based on a configurable weight.
- **Debug Mode:** Optionally display the generated SQL prompt and query for debugging.
- **Local LLM & Embeddings:** Uses a local language model endpoint for chat completions and HuggingFace models for embedding generation.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [LangChain](https://github.com/hwchase17/langchain) and community extensions
- [HuggingFace Transformers and SentenceTransformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss) (e.g., `faiss-cpu`)

Additional dependencies might be required by your local LLM server (e.g., Qwen2.5) and any custom packages from LangChain community.

## Installation

### 1. Clone the Repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Create a Virtual Environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Required Packages:

Create a `requirements.txt` file with the necessary dependencies or install packages individually:

```bash
pip install streamlit pandas sqlalchemy langchain faiss-cpu sentence-transformers
```

> **Note:** You may need to install additional packages depending on your specific environment or LLM configuration.

### 4. Set Up Environment Variables:

Set your API key for the LLM (and any other required credentials). For example, in your shell or within a `.env` file (using `python-dotenv`):

```bash
export OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

### 1. Run the Streamlit App:

```bash
streamlit run table_rag.py
```

### 2. Upload Your Data:

- On the main page, use the file uploader to select a CSV, XLSX, or JSON file.
- The data is ingested, and a SQLite database is built.

### 3. Ask Questions:

- Enter a natural language question in the text input.
- Adjust the **SQL Weight** slider to prioritize SQL-based or semantic retrieval.
- Toggle the **debug** checkbox to see the generated SQL query (for debugging purposes).
- Click **Submit** to see the answers generated via SQL, semantic retrieval, and the combined final answer.

## Configuration

### LLM Settings:
In `app.py`, adjust the `ChatOpenAI` instance parameters (`base_url`, `api_key`, `model`) to match your local LLM server configuration.

This implementation currently uses an LLM served on **LM Studio's server mode**, but any **OpenAI-compatible endpoint** will work.

### Embedding Model:
The code uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace. You can replace this with any compatible model.

### SQL Database:
The database table is always named `training`. You can modify the code to support dynamic table naming if needed.

## Contributing

Contributions and improvements are welcome! Feel free to submit issues or pull requests to enhance functionality, fix bugs, or improve documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the developers of Streamlit, LangChain, and the HuggingFace community for providing amazing tools and libraries.
