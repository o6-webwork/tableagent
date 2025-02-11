# Use official Python image as base
FROM python:3.12.9

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_BASE_URL="http://host.docker.internal:1234/v1"  \
    WORKDIR=/app

# Set the working directory
WORKDIR $WORKDIR

# Copy requirements and install dependencies
COPY requirements.txt $WORKDIR/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . $WORKDIR/

# Expose Streamlit's default port
EXPOSE 8501

# Set entrypoint to run the application
ENTRYPOINT ["streamlit", "run", "table_rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
