# Use slim Python 3.11 to keep image size down
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app
COPY . /app

# Install system dependencies required by TensorFlow and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
RUN pip install -r requirements.txt

# Expose the port uvicorn will run on
EXPOSE 8000

# Run the app (no --reload in production)
CMD ["uvicorn", "app_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]