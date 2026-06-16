FROM python:3.11-slim

WORKDIR /app

# Install git + dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/

COPY params.yaml .
COPY dvc.yaml .
COPY .dvc/ ./.dvc/        

# Initialize git repo (required for DVC)
RUN git init && git config user.email "ci@mlops.com" && git config user.name "CI"

# Expose ports
EXPOSE 8000 7860

# Start FastAPI + Gradio
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app/gradio_app.py"]