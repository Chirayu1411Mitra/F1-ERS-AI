FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static models fastf1_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV FASTF1_ENABLED=0

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]