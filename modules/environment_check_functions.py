'''
CORRECT ENVIRONMENT CHECKING FUNCTIONS:

Applies A* algorithm to determine whether a path exist from defined source and destination fields.
Grid or maze bounds and barriers are taken into account.
'''

import heapq

class Cell:
    def __init__(self):      
        self.parent_i = 0 # row index
        self.parent_j = 0 # column index
        self.f = float('inf') # Total cost of the cell (g + h)    
        self.g = float('inf') # Cost from start to this cell    
        self.h = 0 # Heuristic cost from this cell to destination

def between_limits(row, col, dim):  
    """
    Function that check if the agent is between matrix grid limits,

    Parameters:
        dim (int): Dimension of the matrix (dim x dim).
        row (int): Row position.
        col (int): Column position.

    Returns:
       Boolean: Return whether the agent is between matrix grid limits or not.
    """     
    return (row >= 0) and (row < dim) and (col >= 0) and (col < dim)


def is_barrier(binary_grid, row, col):
    """
    Function that check if the agent is in barrier position or not.

    Parameters:
        binary_grid (matrix): Binary grid matrix representing where are the barriers. Barriers are 
        represented with 1.
        row (int): Row position.
        col (int): Column position.

    Returns:
       Boolean: Return whether the agent is in a barrier or not.
    """     
    return binary_grid[row][col] == 1


def is_destination(row, col, dest):
    """
    Function that determine if the agent is in the destination point.

    Parameters:
        row (int): Row position.
        col (int): Column position.
        dest (tuple): End point position. From (0, dim-1) to (dim-1, dim-1).

    Returns:
       Boolean: Return wether the agent is in  the destination or end point.
    """   
    return row == dest[0] and col == dest[1]

def trace_path(cell_details, dest):
    """
    Function that traces the possible path from source to destination.

    Parameters:
        cell_details (Cell): Cells sum.
        dest (tuple): End point position. From (0, dim-1) to (dim-1, dim-1).

    Returns:
       None: Prinths the path from source to destination if it is possible.
    """   
    print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    for i in path:
        print("->", i, end=" ")
    print()
    
def calculate_h_value(row, col, dest):
    """
    Function that calculates heuristic value. The heuristic cost estimation from neighbor to target.

    Parameters:
        row (int): Row position.
        col (int): Column position.
        dest (tuple): End point position. From (0, dim-1) to (dim-1, dim-1).

    Returns:
       Float: The heuristic cost estimation from neighbor to target.
    """   
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# A* search algorithm
def a_star_search(binary_grid, src, dest, dim, verbose):
    """
    Function that applies A* algorithm to determine whether is possible to go from start point to the 
    destination point. 

    Parameters:
        binary_grid (matrix): Binary grid matrix representing where are the barriers. Barriers are 
        represented with 1.
        src (list): Start point position. From (0, 0) to (dim-1, 0).
        dest (list): End point position. From (0, dim-1) to (dim-1, dim-1).
        dim (int): Dimension of the matrix (dim x dim).
        verbose (Boolean): whether debug print must be shown or not.

    Returns:
       Boolean: Evaluate whether is possible to trace a path from start point to the end.
    """   
    
    # Check if the source and destination are valid
    if not between_limits(src[0], src[1], dim) or not between_limits(dest[0], dest[1], dim):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are blocked
    if is_barrier(binary_grid, src[0], src[1]) or is_barrier(binary_grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(dim)] for _ in range(dim)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(dim)] for _ in range(dim)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Define possible directions (only cardinal directions)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            # If the successor is valid, unblocked, and not visited
            if between_limits(new_i, new_j, dim) and not is_barrier(binary_grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("Path to destination found.")
                    found_dest = True
                    if verbose:
                        # Trace and print the path from source to destination
                        trace_path(cell_details, dest)
                    return found_dest
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")
    return found_dest