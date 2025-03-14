# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Whisper dependencies (e.g., ffmpeg for audio processing)
RUN apt-get update && apt-get install -y ffmpeg

# Pre-download the Whisper model to avoid runtime downloads
RUN python -c "import whisper; whisper.load_model('small')"

# Copy the application code
COPY app/ .

# Expose port 5000
EXPOSE 5000

# Run the app with Gunicorn for production with increased timeout
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]