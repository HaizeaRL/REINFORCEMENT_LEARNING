# REINFORCEMENT_LEARNING

-   **Author**: Haizea Rumayor Lazkano
-   **Last update**: November 2024

------------------------------------------------------------------------

This GitHub project creates different environments based on grid mazes (squares with dimensions dim x dim) with barriers or obstacles along the path. These mazes are designed for an agent to learn the optimal path through **Reinforcement Learning**.

## Overview

The project implements several reinforcement learning algorithms, including **Q-Learning**, **SARSA**, and **Deep Q-Learning**, allowing agents to explore and learn in varying grid maze environments.

## Key Features

- **Maze Environment Creation**: The project offers the ability to create different grid maze environments where an agent can learn. The mazes are generated with random obstacles and barriers to challenge the agent's learning process.
  
- **Heuristic Validation with A\***: Once a maze is generated, it is validated using a heuristic approach. The **A\*** algorithm is applied to ensure that the maze is solvable — i.e., the agent can reach the destination. If the maze is invalid (i.e., no path exists), a new environment is generated.

- **Reinforcement Learning Algorithms**: The project includes Jupyter Notebooks to apply reinforcement learning to the generated mazes using the following algorithms:
  
  - **Q-Learning** and **SARSA**: These algorithms are used for smaller grid environments (ranging from 4x4 to 20x20 grids) to enable the agent to learn basic navigation tasks.
  
  - **Deep Q-Learning**: For larger environments (ranging from 100x100 to 500x500 grids), Deep Q-Learning is employed to allow the agent to learn complex navigation strategies in larger and more challenging mazes.

## Project Structure

The project is organized into the following directories:

- **modules/**: Contains modules with functions used throughout the project.
- **src/**: Contains Python scripts that implement the corresponding Reinforcement Learning algorithms. The script `Reinforcement_learning_with_Q_LEARNING_and_SARSA.ipynb` is for the Q-Learning and SARSA algorithms, while `Reinforcement_learning_with_Deep_Q_LEARNING.ipynb` is for the Deep Q-Learning algorithm.
- **requirements.txt**: Lists the Python packages required to run the project.
- **Dockerfile**: Used to launch the project in a Docker container.

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