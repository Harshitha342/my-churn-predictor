# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app ./app
COPY models ./models

# Expose API port
EXPOSE 8000

# Run the Flask app
CMD ["python", "app/main.py"]
