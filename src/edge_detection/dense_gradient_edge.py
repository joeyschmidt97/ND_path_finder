import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.linalg import norm

def dense_contour_boundary_2D(dataset, boundary_resolution_len=0.02, boundary_mesh_resolution=100, plot=False, edge_margin=0.01):
    """
    Extracts the boundary of regions with 0 or 1 values using contour and interpolates points 
    along the boundary based on a given resolution. Avoids boundary points near the edge of the image.

    Parameters:
    - dataset (xarray.Dataset): The input dataset containing 'points' and 'noise_values'.
    - boundary_resolution_len (float): Desired length between interpolated boundary points.
    - boundary_mesh_resolution (int): Resolution of the meshgrid for contouring.
    - plot (bool): If True, plots the boundary and points.
    - edge_margin (float): The margin to exclude points near the edge of the grid.

    Returns:
    - np.ndarray: Interpolated boundary points.
    """
    # Ensure the required variables are present
    if not {'points', 'noise_values'}.issubset(dataset.variables):
        raise ValueError("Dataset must contain 'points' and 'noise_values' variables.")

    # Extract features and labels
    X = dataset['points'].values  # Shape (num_points, dim)
    y = dataset['noise_values'].values  # Shape (num_points,)

    # Create a grid for contour visualization
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, boundary_mesh_resolution),
                         np.linspace(y_min, y_max, boundary_mesh_resolution))
    
    # Interpolate y values on the grid
    from scipy.interpolate import griddata
    Z = griddata(X, y, (xx, yy), method='linear', fill_value=0.5)  # Interpolate noise values

    # Extract boundary points using contour
    contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=1, colors="black")
    # plt.close()  # Suppress the display

    # Extract boundary points from contour paths
    boundary_points = []
    for path in contours.collections[0].get_paths():
        vertices = path.vertices
        distances = np.sqrt(np.sum(np.diff(vertices, axis=0)**2, axis=1))
        cumulative_distances = np.cumsum(distances)
        cumulative_distances = np.insert(cumulative_distances, 0, 0)

        # Interpolate points along the path
        path_length = cumulative_distances[-1]
        num_points = int(path_length // boundary_resolution_len) + 1
        evenly_spaced_distances = np.linspace(0, path_length, num=num_points)

        interp_fn = interp1d(cumulative_distances, vertices, axis=0, kind='linear')
        interpolated_points = interp_fn(evenly_spaced_distances)

        # Filter out points close to the edge (using the margin)
        valid_points = []
        for point in interpolated_points:
            if (point[0] > x_min + edge_margin and point[0] < x_max - edge_margin and
                point[1] > y_min + edge_margin and point[1] < y_max - edge_margin):
                valid_points.append(point)

        boundary_points.append(np.array(valid_points))

    # Combine all boundary segments into a single array
    boundary_points = np.vstack(boundary_points)

    # Plotting if required
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='red', s=5, label='Boundary Points')
        plt.legend()
        plt.title("Boundary Detection with Contour")
        plt.show()

    return boundary_points
