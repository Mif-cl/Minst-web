# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY scr ./scr
COPY checkpoint.pt .
COPY .streamlit .streamlit

# Set Streamlit to run on external port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "scr/app.py", "--server.enableCORS=false", "--server.port=8501", "--server.address=0.0.0.0"]