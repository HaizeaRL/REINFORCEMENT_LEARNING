import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import sys

# Add the parent directory (where modules is located) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import environment_check_functions as echf

def visualize_matrix (dim, grid_matrix):
    """
    Function that visualize created grid maze positions.

    Parameters:
        dim (int): Dimension of the matrix. dim x dim.
        grid_matrix (matrix): Grid matrix positions (0,0) to (dim-1, dim-1).

    Returns:
       None: visualize created grid maze positions.
    """   
    for row in range(0, dim):
        print(grid_matrix[row])


def get_all_grid_matrix_position (dim, grid_matrix):
    """
    Function that gets all grid matrix positions in a list.

    Parameters:
        dim (int): Dimension of the matrix. dim x dim.
        grid_matrix (matrix): Grid matrix positions (0,0) to (dim-1, dim-1).

    Returns:
       list: all grid matrix positions in a list.
    """   
    positions= []
    for row in range(0, dim):
        for col in range(0, dim):
            positions.append(grid_matrix[row][col])
    return positions

def determine_num_barriers (all_grid_matrix_positions):
    """
    Function to determine the number of barrier positions.

    Parameters:
        all_grid_matrix_positions (list): A list of all grid matrix positions, ranging from (0, 0) to (dim-1, dim-1).

    Returns:
        int: The number of barriers to create. Calculated as one quarter of the total positions, rounded to the nearest 
        integer.
    """   
    total_positions = len(all_grid_matrix_positions)
    return round(total_positions / 4) # quarter 

def get_barriers_positions(all_grid_matrix_positions):
    """
    Function to determine the barrier's positions.

    Parameters:
        all_grid_matrix_positions (list): A list of all grid matrix positions, ranging from (0, 0) to (dim-1, dim-1).

    Returns:
        int: The number of barriers created. Calculated as one quarter of the total positions, rounded to the nearest 
        integer.
        list: List of barriers positions.
    """
    # determine how many barriers
    barriers =[]
    n_barriers =  determine_num_barriers(all_grid_matrix_positions)

    # get barriers position
    count= 0   
    while count < n_barriers:
        # choose randomly barriers position
        pos = random.randint(0, len(all_grid_matrix_positions)-1)
        
        # add barrier position without repetition.
        if all_grid_matrix_positions[pos] not in barriers:           
            barriers.append(all_grid_matrix_positions[pos])
            count+=1
            
    return n_barriers, barriers

def visualize_environment(dim, barriers, start_pos, end_pos):
    """
    Function to visualize the final matrix grid (maze) or agent environment.
    Barriers are represented with black patches, the start point in green, and the end point in red.

    Parameters:
        dim (int): Dimension of the matrix (dim x dim).
        barriers (list): List of barrier positions.
        start_pos (tuple): Start position in the maze.
        end_pos (tuple): End position in the maze.

    Returns:
        None: Displays the final matrix grid (maze), showing barriers, start, and end positions.
        The agent environment.
    """       
    # Plot grid
    fig, ax = plt.subplots(figsize=(4, 4))

    # Set the grid's dimensions
    for x in range(dim + 1):
        ax.plot([x, x], [0, dim], color="black")  # Vertical lines
        ax.plot([0, dim], [x, x], color="black")  # Horizontal lines
        
        
    # Draw barriers
    for (row, col) in barriers:
        # Note that `col` is x-position, and `row` is y-position
        ax.add_patch(patches.Rectangle((col, dim - 1 - row), 1, 1, color="black"))
        
    # Draw black squares at the start and end positions
    ax.add_patch(patches.Rectangle((start_pos[1], dim - 1- start_pos[0]), 1, 1, color="green"))
    ax.add_patch(patches.Rectangle((end_pos[1], dim - 1- end_pos[0]), 1, 1, color="red"))

    # Set the limits of the plot
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dim)

    # Hide the axes
    ax.axis("off")

    # Show the plot
    plt.show()

def get_start_point(dim, grid_matrix):
    """
    Function that determine start point of the maze.

    Parameters:
        dim (int): Dimension of the matrix. dim x dim.
        grid_matrix (matrix): Grid matrix positions (0,0) to (dim-1, dim-1).

    Returns:
       Tuple: Start point of the maze. Forced to be in first column.
    """   
    # get randomly start position row
    row = random.randint(0, dim-1)
    
    return grid_matrix[row][0]

def get_end_point(dim, grid_matrix):
    """
    Function that determine end point of the maze.

    Parameters:
        dim (int): Dimension of the matrix. dim x dim.
        grid_matrix (matrix): Grid matrix positions (0,0) to (dim-1, dim-1).

    Returns:
       Tuple: End point of the maze. Forced to be in last column.
    """   
     # get randomly end position row
    row = random.randint(0, dim-1)
    
    return grid_matrix[row][dim-1]  

def is_end_point(row, col, dest):
    """
    Function that determine if is the end point.

    Parameters:
        row (int): Row position.
        col (int): Column position.
        dest (tuple): End point position. From (0, dim-1) to (dim-1, dim-1).

    Returns:
       Boolean: Return wether is in the destination or end point.
    """   
    return row == dest[0] and col == dest[1]

def is_start_point(row, col, start):
    """
    Function that determine if is the end point.

    Parameters:
        row (int): Row position.
        col (int): Column position.
        start (tuple): Start point position. From (0, 0) to (dim-1, 0).

    Returns:
       Boolean: Return wether is in start point or not.
    """   
    return row == start[0] and col == start[1]

def get_binary_grid_matrix (dim, barriers, start_pos, end_pos):
    """
    Function that obtain binary grid matrix. Barriers are represented as 1.

    Parameters:
        dim (int): Dimension of the matrix (dim x dim).
        barriers (list): List of barrier positions.
        start_pos (tuple): Start position in the maze.
        end_pos (tuple): End position in the maze.

    Returns:
       Matrix: Return binary grid matrix representing where are the barriers. Barriers are 
        represented with 1.
    """   
    # initialize matrix
    binary_matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # mark barriers as 1
    for row, col in barriers:
        if not is_end_point(row, col, end_pos) and not is_start_point(row, col, start_pos):
            binary_matrix[row][col] = 1      
    return binary_matrix

def create_env_reward_matrix(dim, barriers, src, dest):
    """
    Function that obtain binary grid matrix. Barriers are represented as 1.

    Parameters:
        dim (int): Dimension of the matrix (dim x dim).
        barriers (list): List of barrier positions.
        src (tuple): Start position in the maze.
        dest (tuple): End position in the maze.

    Returns:
       Matrix: Reward matrix to apply Reinforcement Learning in it.
    """   
    # initialize matrix
    m = [[0 for _ in range(dim)] for _ in range(dim)]

    # mark barriers as -10
    for row, col in barriers:
        if not is_end_point(row, col, dest) and not is_start_point(row, col, src):
            m[row][col] = -10 
            
    # mark destination as 100
    m[dest[0]][dest[1]] = 100
    return m

def create_environment(verbose = False):
    """
    Function to create a randomly generated matrix grid (environment) for an agent.
    Creates a matrix grid with barriers and evaluates whether a valid path is possible 
    from a randomly set start point to the end point.

    Parameters:
        verbose (Boolean): whether debug print must be shown or not.

    Returns:
        tuple: Returns the environment's dimension, grid matrix positions, barriers,
               start and end positions, and the binary grid representing the barriers.
    """   
    while True:
        print("Creating maze")

        # Get maze dimension randomly
        dim = random.randint(4,20)
        print(f"Grid of: {dim} x {dim}")

        # Get grids positions
        grid_matrix_positions = [[(row, col) for col in range(0, dim)] for row in range(0, dim)]

        # Obtain barriers 
        grid_positions = get_all_grid_matrix_position(dim, grid_matrix_positions)
        n_barriers, barriers = get_barriers_positions(grid_positions)
        print(f"Barriers count: {n_barriers}")

        # Obtain start and end position
        src_position = get_start_point(dim, grid_matrix_positions)
        dest_position = get_end_point(dim, grid_matrix_positions)
        print(f"Source: {src_position}")
        print(f"Destination: {dest_position}")

        # get binary grid determining where are the barriers
        binary_grid = get_binary_grid_matrix (dim, barriers, src_position, dest_position)

        # Check for a valid path using the A* algorithm
        if echf.a_star_search(binary_grid, src_position, dest_position, dim, verbose):
            reward_matrix = create_env_reward_matrix(dim, barriers, src_position, dest_position)
            break
           
    # return resulted environment data
    return dim, grid_matrix_positions, barriers, src_position, dest_position, reward_matrix