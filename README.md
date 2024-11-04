# REINFORCEMENT_LEARNING

## Installation and Run Steps

To get started with this project, please follow these steps:

### Prerequisites

1. **Install Docker**: Ensure Docker is installed on your machine. You can download and install it from the [Docker website](https://www.docker.com/products/docker-desktop).

### Configuration for Matplotlib

To use Matplotlib in a Docker container with GUI support, you'll need to configure X11 forwarding on your Windows machine. Follow these instructions:

1. **Install XLaunch**:
   - Download and install **XLaunch** from the [Xming website](https://sourceforge.net/projects/xming/).
   - Launch XLaunch and choose **"Multiple windows"** when prompted.
   - Set the display number to **0**.
   - Select **"Start No client"**.
   - Choose **"Native OpenGL"**.
   - Check **"No access control"** to allow connections.

2. **Get Your Windows IP Address**:
   - Open Command Prompt and run the following command to find your IP address:
     ```bash
     ipconfig
     ```
   - Note the IPv4 Address (e.g., `192.168.1.100`).

### Building the Docker Image

1. **Navigate to the Project Directory**:
   Open your terminal and navigate to the root directory of the project where the `Dockerfile` is located.

2. **Build the Docker Image**:
   Run the following command to build the Docker image. Replace `<app>` with corresponding applcation name:
   ```bash
   docker build -t <app> .
   ```
2. **Run the Docker Image**:
    Replace `<ip_address>` and `<app>` with corresponding values:
    ```bash
    docker run --rm -it --env=DISPLAY=<ip_address>:0 -v="$(Get-Location):/app" <app>
    ```
3. **Navigate to corresponding script and run the script**:
   Run the `main.py` script as many times as needed to generate recommendations:
   ```bash
    cd src
    python main.py
   ```