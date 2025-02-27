import numpy as np

def coords_to_index(grid,coords):
    index = []
    for (x, y) in coords:
        index.append(int(grid[y][x]))
    return index

def index_to_coords_indexes(coords, grid_width, grid_height):
    indexes = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    for i, (x, y) in enumerate(coords):
        indexes[y][x] = i
    return np.array(indexes).flatten()

def s_curve(grid):
    rows = len(grid)
    cols = len(grid[0])
    order = []
    for y in range(rows):
        if y % 2 == 0:
            # Left-to-right for even rows
            order.extend((x, y) for x in range(cols))
        else:
            # Right-to-left for odd rows
            order.extend((x, y) for x in reversed(range(cols)))
    return order
    
def compute_curve_order(grid, orientation):
    if orientation == 's':
        order = s_curve(grid)
    elif orientation == 'sr':
        order = s_curve(grid)   
        order = [(y,x) for x,y in order]       
    return order
