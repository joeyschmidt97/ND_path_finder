import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from src.data_generators.perlin_generator import PerlinNoiseGenerator # Import your class here

# Initialize PerlinNoiseGenerator
generator = PerlinNoiseGenerator(res=[50, 50], octv=3, seed=42)
perlin_dataset = generator.perlin_data_xarray(noise_rescaling=[0, 1])
binary_data = (perlin_dataset['noise_values'].values > 0.5).astype(int)

# Edge detection parameters
sparsity_factor = 0.4  # Fraction of points to retain
min_neighbors = 2
avg_neighbors = 25
threshold_ratio = 0.15

# Make data sparse by selecting a subset of points
sparse_points = []
grid_size = binary_data.shape[0]
for x in range(grid_size):
    for y in range(grid_size):
        if np.random.rand() < sparsity_factor:  # Keep only a fraction of points
            sparse_points.append((x, y, binary_data[x, y]))


# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Calculate max of 2-neighbor distances and average 25-neighbor distance across all points
two_neighbor_distances = []
five_neighbor_averages = []

for i, (x, y, label) in enumerate(sparse_points):
    # Find distances to all other points
    point_distances = [euclidean_distance((x, y), (nx, ny)) for j, (nx, ny, nlabel) in enumerate(sparse_points) if
                       i != j]
    point_distances.sort()

    # Store second nearest (for 2-neighbor max) and mean of 25 nearest (for 25-neighbor avg)
    two_neighbor_distances.append(point_distances[min_neighbors - 1])  # Second closest neighbor
    five_neighbor_averages.append(np.mean(point_distances[:avg_neighbors]))  # Average distance of 25 nearest

# Set d_threshold as the greater of the two metrics
max_2_neighbors = max(two_neighbor_distances)
avg_5_neighbors = np.mean(five_neighbor_averages)
if max_2_neighbors > avg_5_neighbors:
    d_threshold = max_2_neighbors
    print(f"Using maximum distance {d_threshold} for {min_neighbors} minimum neighbors as d_threshold.")
else:
    d_threshold = avg_5_neighbors
    print(f"Using average distance of {d_threshold} for {avg_neighbors} neighbors as d_threshold.")

# Calculate gradients for sparse data with the dynamic d_threshold
gradients = []
gradient_magnitudes = []
for i, (x, y, label) in enumerate(sparse_points):
    gradient_sum = np.array([0.0, 0.0])  # Sum of gradients for each point
    num_neighbors = 0

    for j, (nx, ny, nlabel) in enumerate(sparse_points):
        if i != j:
            distance = euclidean_distance((x, y), (nx, ny))
            if distance <= d_threshold:
                direction = np.array([nx - x, ny - y]) / distance
                weight = 1 / distance
                gradient = (nlabel - label) * direction * weight
                gradient_sum += gradient
                num_neighbors += 1

    if num_neighbors > 0:
        gradient_avg = gradient_sum / num_neighbors
    else:
        gradient_avg = np.array([0.0, 0.0])

    gradient_magnitude = np.linalg.norm(gradient_avg)
    gradient_magnitudes.append(gradient_magnitude)
    gradients.append((x, y, gradient_avg[0] * 10, gradient_avg[1] * 10, label))

# Define a threshold for boundary detection
gradient_threshold = threshold_ratio * max(gradient_magnitudes)

# Identify boundary points based on gradient magnitude and label
boundary_points = [(x, y, label) for (x, y, gx, gy, label), mag in zip(gradients, gradient_magnitudes) if
                   mag > gradient_threshold]

# Plotting instructions for sparse data with gradients and boundary points
plt.figure(figsize=(10, 10))
plt.imshow(binary_data, cmap='gray', origin='lower', alpha=0.3)  # Background Perlin noise

# Plot gradient vectors
for (x, y, gx, gy, label) in gradients:
    plt.scatter(y, x, color='blue')  # Plot sparse data points
    plt.arrow(y, x, gy, gx, head_width=0.3, head_length=0.3, fc='red', ec='red')

# Highlight boundary points with different colors based on label
for (x, y, label) in boundary_points:
    color = 'green' if label == 0 else 'yellow'
    plt.scatter(y, x, color=color, edgecolor='black', s=100)

plt.title("Sparse Binary Data with Gradient Vectors and Boundary Points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(False)
plt.show()
