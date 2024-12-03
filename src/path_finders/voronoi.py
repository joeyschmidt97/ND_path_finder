import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d



def voronoi_points(coord_array, vals, threshold, plot=False):

    filtered_coord = []
    filtered_vals = []
    threshold = 0.5

    for ind,v in enumerate(vals):
        if v > threshold:
            filtered_vals.append(v)

            temp_coord = [coord[ind] for coord in coord_array]
            filtered_coord.append(temp_coord)

    # Compute Voronoi diagram
    vor = Voronoi(filtered_coord)
    
    if plot:
        # Plotting
        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        # ax.plot(filtered_coord[:, 0], filtered_coord[:, 1], 'ko')
        
        # Configure the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()









##From here starts Octavio's contribution
def minkowski_distance(p1, p2, p_value):
    return np.sum(np.abs(p1 - p2) ** p_value, axis=-1) ** (1 / p_value)


def voronoi_points_minkowski(coord_array, vals, resolution, parameter_p, x_range=None, y_range=None, plot=False):

    filtered_coord = []
    filtered_vals = []      
    threshold = 0.5

    for ind,v in enumerate(vals):
        if v > threshold:
            filtered_vals.append(v)

            temp_coord = [coord[ind] for coord in coord_array]
            filtered_coord.append(temp_coord)

    if x_range is None:
        x_min, x_max = filtered_coord[:, 0].min() - 1, filtered_coord[:, 0].max() + 1
        x_range = (x_min, x_max)
    if y_range is None:
        y_min, y_max = filtered_coord[:, 1].min() - 1, filtered_coord[:, 1].max() + 1
        y_range = (y_min, y_max)

    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])
    # Determine the closest generator for each grid point based on Minkowski distance
    regions = np.zeros(len(grid_points), dtype=int)
    for i, grid_point in enumerate(grid_points):
        distances = minkowski_distance(filtered_coord, grid_point, parameter_p)
        regions[i] = np.argmin(distances)

    # Reshape the regions into a 2D array for plotting
    regions_reshaped = regions.reshape(resolution, resolution)
    
    # Plot the Voronoi diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(regions_reshaped, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), 
              origin='lower', cmap='tab20', alpha=0.5)
    ax.scatter(filtered_coord[:, 0], filtered_coord[:, 1], color='red', zorder=5, label='Generating Points')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(f"Voronoi Diagram with Minkowski Distance (p={parameter_p})")
    ax.legend()
    plt.grid(True)
    plt.show()