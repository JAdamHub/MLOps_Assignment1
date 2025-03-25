# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and data
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the prediction script when the container starts
CMD ["python", "src/prediction.py"]