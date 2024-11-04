# Use the official Python 3.7 image from Docker Hub
FROM python:3.7

# Set the working directory inside the container
WORKDIR /usr/local/app

# Install system-level dependencies including tkinter for matplotlib
RUN apt-get update && apt-get install -y \
    python3-tk \
    libxrender1 \
    libxext6 \
    libsm6 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set environment variable to make Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Set the default command to start a bash shell
CMD ["/bin/bash"]
