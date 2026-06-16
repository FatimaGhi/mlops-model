FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/

COPY params.yaml .
COPY dvc.yaml .
COPY .dvc/ ./.dvc/        
COPY data/.gitignore ./data/.gitignore


# Expose ports
EXPOSE 8000 7860

# Start FastAPI + Gradio
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app/gradio_app.py"]