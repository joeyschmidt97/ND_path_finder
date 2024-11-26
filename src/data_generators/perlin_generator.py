import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from perlin_noise import PerlinNoise
from joblib import Parallel, delayed



class PerlinNoiseGenerator:

    def __init__(self, res, octv, seed=None, time_test=False):
        """
        Initializes the PerlinNoiseGenerator with resolution, octaves, and optional seed.

        Args:
        - resolution (list[int]): Resolution along each dimension.
        - octaves (int): Number of octaves for Perlin noise.
        - seed (int, optional): Seed for reproducibility.
        """
        self.seed = seed
        self.octaves = octv
        self.resolution = res
        self.noise = PerlinNoise(octaves=self.octaves, seed=self.seed)  # Pre-create noise object for efficiency
        self.time_test = time_test
        self.xarray = None



    def plot_2D_xarray(self, xdataset=None):

        if xdataset is None:
            xdataset = self.xarray
            if xdataset is None:
                xdataset = self.perlin_data_xarray()

        # Extract points and noise values
        points = xdataset['points'].values  # 2D array of shape (num_points, 2)
        noise_values = xdataset['noise_values'].values  # 1D array of shape (num_points,)

        # Scatter plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            points[:, 0],  # x0 values
            points[:, 1],  # x1 values
            c=noise_values,  # Color based on noise values
            cmap='viridis',
            s=5,  # Size of points
            alpha=0.8  # Transparency
        )
        plt.colorbar(scatter, label='Noise Value')
        plt.title("Scatter Plot of 2D Perlin Noise")
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.show()




    def perlin_data_xarray(self, noise_rescaling=[0, 1], noise_cutoff_list=[0.5, 0, 1]):
        """
        Generates an N-dimensional Perlin noise dataset with normalized features, using xarray.

        Args:
        - noise_rescaling (list[float]): Range to rescale noise values, default is [0, 1].
        - noise_cutoff_list (list[float], optional): List of cutoff values [divide, bottom, top].

        Returns:
        - xarray.Dataset: Dataset containing noise values, normalized features, and metadata.
        """
        # Measure time for Perlin noise generation
        perlin_matrix = self._ND_perlin_matrix(noise_rescaling, noise_cutoff_list)
        
        # Measure time for creating an array of coordinates
        if self.time_test: start_time = time.time()
        grid = np.meshgrid(*[np.linspace(0, 1, dim_res) for dim_res in self.resolution], indexing="ij")
        points = np.stack(grid, axis=-1).reshape(-1, len(self.resolution))  # Flatten to (num_points, dimensions)
        noise_values = perlin_matrix.flatten()  # Flatten noise values to match the coordinate tuples
        if self.time_test: coordinate_creation_time = time.time() - start_time

        # Measure time for creating xarray.Dataset
        if self.time_test: start_time = time.time()
        dataset = xr.Dataset(
            {
                'noise_values': (['points'], noise_values),
            },
            coords={
                'points': (['points', 'dim'], points),  # Use a 2D array for coordinates
            },
            attrs={
                'octaves': self.octaves, 
                'resolution': self.resolution,
                'dimensions': len(self.resolution)
            }
        )
        if self.time_test: dataset_creation_time = time.time() - start_time

        # Print timing results
        if self.time_test: 
            print("Time for creating coordinates:", coordinate_creation_time, "seconds")
            print("Time for xarray.Dataset creation:", dataset_creation_time, "seconds")

        self.xarray = dataset

        return dataset



    def _ND_perlin_matrix(self, noise_rescaling=[0, 1], noise_cutoff_list=[0.5,0,1]):
        noise_min, noise_max = noise_rescaling

        # Generate mesh grid for each dimension
        if self.time_test: start_time = time.time()
        grid = np.meshgrid(*[np.linspace(0, 1, res) for res in self.resolution], indexing='ij')
        coords = np.stack(grid, axis=-1).reshape(-1, len(self.resolution))
        if self.time_test: grid_time = time.time() - start_time



        # Parallel computation using joblib for noise generation
        if self.time_test: start_time = time.time()
        noise_values = Parallel(n_jobs=-1)(delayed(self.noise)(tuple(coord)) for coord in coords)  # -1 uses all available cores
        noise_values = np.array(noise_values)
        noise_values = noise_min + (noise_values + 1) * (noise_max - noise_min) / 2
        noise_values = noise_values.reshape(self.resolution)
        if self.time_test: noise_rescaling_time = time.time() - start_time


        # Apply cutoffs if needed
        if noise_cutoff_list:
            if self.time_test: start_time = time.time()
            noise_divide, noise_bottom, noise_top = noise_cutoff_list
            noise_values = np.where(noise_values < noise_divide, noise_bottom, noise_top)
            if self.time_test: 
                cutoff_time = time.time() - start_time
                print("Time for applying cutoffs:", cutoff_time, "seconds")

        if self.time_test: 
            print("Time for generating mesh grid and coordinates:", grid_time, "seconds")
            print("Time for parallel noise computation and rescaling:", noise_rescaling_time, "seconds")

        # print(noise_values[0:10])

        return noise_values

