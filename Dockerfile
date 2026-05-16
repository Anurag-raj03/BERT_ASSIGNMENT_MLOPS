# Use the slim python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional, depending on python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model during build phase to cache it in the image.
# This ensures that the container startup time is extremely fast.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='bvanaken/clinical-assertion-negation-bert')"

# Copy the rest of the application
COPY app/ app/

# Expose the port Cloud Run uses
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
