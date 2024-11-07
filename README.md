# REINFORCEMENT_LEARNING

-   **Author**: Haizea Rumayor Lazkano
-   **Last update**: November 2024

------------------------------------------------------------------------

## Installation

To ensure a clean and isolated environment for this project, a `Dockerfile` is provided to launch Docker locally and run a Python 3.7 container. This approach ensures that the local environment remains unaffected and that all dependencies are installed within the container.

### Prerequisites:

- **Docker** must be installed on your machine. You can download and install Docker from the [official website](https://www.docker.com/get-started).

### Steps to Set Up and Run the Project:

1. **Build the Docker Image**:

   First, build the Docker image for the project using the provided `Dockerfile`. Replace `<app-name>` with the name of your application or service:
   ```bash
   docker build -t <app-name> .
   ```

2. **Run the Docker Container**:

   After building the image, run the container using the following command:
   ```bash
   docker run -it --rm -p 8888:8888 -v ${PWD}:/app <app-name>
   ```

  This command will:

- Mount your current directory (`${PWD}`) to the `/app` folder inside the container.
- Map port `8888` from the container to your local machine, allowing access to Jupyter.
- Automatically remove the container when it stops (`--rm`).


3. **Access Jupyter Notebooks**:

   Once the container is running, you can access the Jupyter notebook interface by opening your browser and navigating to `http://localhost:8888`.


If you execute the scripts presented in the project, the results will be created and saved in the container. If you want to retrieve the results locally, follow these instructions:

**Retrieve resuts locally**   
   
   Identify the running Docker container:
   ```bash
    docker ps
   ```

   Navigate to the project directory and copy the file from the container to your local machine, replacing `<docker_instance>` and `<docker_path_to_retrieve>` with the corresponding values.
   ```bash
    docker cp <docker_instance>:/<docker_path_to_retrieve> .
   ```