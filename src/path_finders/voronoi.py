import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

def compute_sink_vertices(boundary_points, gradient_vectors, plot=False):
    """
    Computes the Voronoi diagram based on boundary points and gradient vectors, 
    identifies vertices where all gradients point inward, and optionally plots the result.
    
    Parameters:
    - boundary_points: A list of 2D boundary points that define the Voronoi diagram.
    - gradient_vectors: A list of gradient vectors corresponding to the boundary points.
    - plot (optional): A boolean flag to indicate whether to plot the results. Default is False.
    
    Returns:
    - midpath_coords: The coordinates of Voronoi vertices where all gradients point inward.
    """
    
    # Combine boundary points and gradient vectors into single arrays
    combined_boundary_points = np.vstack([np.vstack(segment) for segment in boundary_points])
    combined_gradients = np.vstack([np.vstack(grad) for grad in gradient_vectors])

    # Compute the Voronoi diagram based on the combined boundary points
    vor = Voronoi(combined_boundary_points)

    # Initialize a set to store vertices to highlight
    highlight_vertices = set()

    # Loop through all Voronoi vertices
    for vertex_idx, vertex in enumerate(vor.vertices):
        # Skip vertices that are outside the bounds (x, y should be between 0 and 1)
        if np.any(vertex < 0) or np.any(vertex > 1):
            continue

        inward_count = 0
        outward_count = 0

        # Loop through each region (site) associated with the current vertex
        for point_idx in range(len(vor.point_region)):
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]

            # Skip regions with infinite vertices and check if the vertex belongs to the region
            if -1 in region or vertex_idx not in region:
                continue

            # Compute the vector from the site to the vertex
            site_to_vertex = vertex - combined_boundary_points[point_idx]

            # Check the direction of the gradient vector relative to the site-to-vertex vector
            if np.dot(combined_gradients[point_idx], site_to_vertex) < 0:
                inward_count += 1  # Gradient points inward (towards the vertex)
            else:
                outward_count += 1  # Gradient points outward (away from the vertex)

        # If all gradients point inward, highlight the vertex
        if outward_count == 0:
            highlight_vertices.add(vertex_idx)

    # Extract the coordinates of highlighted vertices
    midpath_coords = vor.vertices[list(highlight_vertices)]

    # Plot the Voronoi diagram with highlighted vertices if requested
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)

        # Highlight vertices where all gradients point inward
        if len(midpath_coords) > 0:
            plt.scatter(midpath_coords[:, 0], midpath_coords[:, 1], color='g', label='Highlighted Vertices')
        
        # Configure plot appearance
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    return midpath_coords


def voronoi_points(boundary_points, gradient_vectors, plot=False):
    """
    Computes and optionally plots the Voronoi diagram for the given boundary points and gradient vectors.
    
    Parameters:
    - boundary_points: A list of 2D boundary points that define the Voronoi diagram.
    - gradient_vectors: A list of gradient vectors corresponding to the boundary points.
    - plot (optional): A boolean flag to indicate whether to plot the results. Default is False.
    
    Returns:
    - vor: The Voronoi diagram object containing the Voronoi regions and vertices.
    """
    
    # Combine boundary points and gradient vectors into single arrays
    combined_boundary_points = np.vstack([np.vstack(segment) for segment in boundary_points])
    combined_gradients = np.vstack([np.vstack(grad) for grad in gradient_vectors])

    # Compute the Voronoi diagram based on the combined boundary points
    vor = Voronoi(combined_boundary_points)

    # Plot the Voronoi diagram and gradient vectors if requested
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)

        # Add gradient vectors to the plot as red arrows
        plt.quiver(
            combined_boundary_points[:, 0], combined_boundary_points[:, 1],
            combined_gradients[:, 0] * 0.005, combined_gradients[:, 1] * 0.005,
            angles='xy', scale_units='xy', scale=0.01, color='r', alpha=0.8, headwidth=3
        )
        
        # Configure plot appearance
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    return vor
