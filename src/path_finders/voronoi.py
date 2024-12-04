import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


def compute_sink_vertices(boundary_points, gradient_vectors, plot=False):

    combined_boundary_points = np.vstack([np.vstack(segment) for segment in boundary_points])
    combined_gradients = np.vstack([np.vstack(grad) for grad in gradient_vectors])

    # Compute the Voronoi diagram
    vor = Voronoi(combined_boundary_points)

    # Initialize a set to store vertices to highlight
    highlight_vertices = set()

    # Loop through Voronoi vertices
    for vertex_idx, vertex in enumerate(vor.vertices):
        if np.any(vertex < 0) or np.any(vertex > 1):
            # Skip vertices outside the diagram bounds
            continue

        inward_count = 0
        outward_count = 0

        # Check all regions (sites) associated with the vertex
        for point_idx in range(len(vor.point_region)):
            # Get the region of the point
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]

            # Skip regions with infinite vertices
            if -1 in region or vertex_idx not in region:
                continue

            # Compute vector from site to vertex
            site_to_vertex = vertex - combined_boundary_points[point_idx]

            # Check if gradient points "into" or "away" from the vertex
            if np.dot(combined_gradients[point_idx], site_to_vertex) < 0:
                inward_count += 1  # Gradient points into the vertex
            else:
                outward_count += 1  # Gradient points away from the vertex

        # Retain vertex if all gradients point inward
        if outward_count == 0:
            highlight_vertices.add(vertex_idx)

    # Extract the coordinates of highlighted vertices
    midpath_coords = vor.vertices[list(highlight_vertices)]

    # Plotting the results
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        plt.quiver(
            combined_boundary_points[:, 0], combined_boundary_points[:, 1],
            combined_gradients[:, 0] * 0.005, combined_gradients[:, 1] * 0.005,
            angles='xy', scale_units='xy', scale=0.01, color='r', alpha=0.8, headwidth=3
        )
        # Highlight vertices
        if len(midpath_coords) > 0:
            plt.scatter(midpath_coords[:, 0], midpath_coords[:, 1], color='g', label='Highlighted Vertices')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    return midpath_coords

# Example usage
# midpath_coords = compute_sink_vertices(boundary_points, gradients, plot=True)




def voronoi_points(boundary_points, gradient_vectors, plot=False):

    combined_boundary_points = np.vstack([np.vstack(segment) for segment in boundary_points])
    combined_gradients = np.vstack([np.vstack(grad) for grad in gradient_vectors])

    # Compute Voronoi diagram
    vor = Voronoi(combined_boundary_points)

    if plot:
        # Plotting
        # plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
        plt.quiver(
            combined_boundary_points[:, 0], combined_boundary_points[:, 1],
            combined_gradients[:, 0]*.005, combined_gradients[:, 1]*.005,
            angles='xy', scale_units='xy', scale=0.01, color='r', alpha=0.8, headwidth=3
        )
        # ax.plot(filtered_coord[:, 0], filtered_coord[:, 1], 'ko')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Configure the plot
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()

    return vor

















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