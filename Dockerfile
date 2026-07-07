# MedSim — single image that can run the Streamlit UI or the FastMCP API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the derived data artifacts & sanity-check the catalog at build time.
RUN python scripts/initialize_data.py

EXPOSE 8501 8000

# Default: launch the UI. Override the command to run the API instead:
#   docker run ... python run_api.py
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
