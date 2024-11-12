# Use the official TensorFlow 2.10.0 image for CPU from Docker Hub
FROM tensorflow/tensorflow:2.10.0

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

# Suppress TensorFlow messages
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1 

# Command to run Jupyter notebook in the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
