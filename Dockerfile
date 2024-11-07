# Use the official Python 3.7 image from Docker Hub
FROM python:3.7

# Set the working directory inside the container
WORKDIR /usr/local/app

# Install basic dependencies (excluding X11 libraries and tkinter)
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the required packages jupyter and from requirements.txt
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set environment variable to make Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Command to run Jupyter notebook in the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
