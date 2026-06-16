FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY dvc.yaml .
COPY .dvc/ ./.dvc/
COPY data/churn.csv.dvc ./data/churn.csv.dvc

RUN git init && git config user.email "ci@mlops.com" && git config user.name "CI"


COPY params.yaml .


# Expose ports
EXPOSE 8000 7860

# Start FastAPI + Gradio
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app/gradio_app.py"]