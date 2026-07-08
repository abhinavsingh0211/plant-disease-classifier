# Container for the Streamlit demo.
# Build:  docker build -t plant-disease-app .
# Run:    docker run -p 8501:8501 plant-disease-app
FROM python:3.11-slim

# System deps kept minimal; TF wheels are self-contained.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install the lightweight (CPU) app dependencies first for better layer caching.
COPY app/requirements.txt ./app-requirements.txt
RUN pip install -r app-requirements.txt

# Copy the code and the trained artefacts.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py"]
