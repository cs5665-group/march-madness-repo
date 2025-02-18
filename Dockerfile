# Use the official Python Alpine image
FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for PyTorch & Pandas
RUN apk add --no-cache \
    gcc g++ musl-dev \
    libffi-dev openssl-dev \
    lapack-dev blas-dev \
    py3-numpy py3-scipy

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose a port (optional, if running a service)
EXPOSE 8000

# Default command to run your script
CMD ["python", "main.py"]
