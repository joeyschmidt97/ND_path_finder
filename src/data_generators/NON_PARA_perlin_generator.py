import time
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from perlin_noise import PerlinNoise


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

    def perlin_data_xarray(self, noise_rescaling=[0, 1], noise_cutoff_list=None):
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
        
        # Measure time for converting to np.array and creating xarray DataArray
        if self.time_test: start_time = time.time()
        coords = {f"x{i}": np.linspace(0, 1, dim_res) for i, dim_res in enumerate(self.resolution)}
        data_array = xr.DataArray(np.array(perlin_matrix), dims=coords.keys(), coords=coords)
        if self.time_test: xarray_conversion_time = time.time() - start_time

        # Measure time for generating normalized feature coordinates
        if self.time_test: start_time = time.time()
        indices = np.indices(data_array.shape).reshape(len(self.resolution), -1).T
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(indices)
        if self.time_test: normalization_time = time.time() - start_time

        # Measure time for creating xarray.Dataset
        if self.time_test: start_time = time.time()
        dataset = xr.Dataset(
            {
                'noise_values': data_array,
                'normalized_features': (['points', 'coordinates'], normalized_features),
            },
            coords={
                'points': np.arange(len(normalized_features)),
                'coordinates': [f'x{i}' for i in range(len(self.resolution))],
            },
            attrs={'octaves': self.octaves, 
                   'resolution': self.resolution,
                   'dimensions': len(self.resolution)}
        )
        if self.time_test: dataset_creation_time = time.time() - start_time

        # Print timing results
        if self.time_test: 
            print("Time for converting to xarray DataArray:", xarray_conversion_time, "seconds")
            print("Time for feature normalization:", normalization_time, "seconds")
            print("Time for xarray.Dataset creation:", dataset_creation_time, "seconds")

        return dataset

    def _ND_perlin_matrix(self, noise_rescaling=[0, 1], noise_cutoff_list=None):
        noise_min, noise_max = noise_rescaling

        # Generate mesh grid for each dimension
        if self.time_test: start_time = time.time()
        grid = np.meshgrid(*[np.linspace(0, 1, res) for res in self.resolution], indexing='ij')
        coords = np.stack(grid, axis=-1).reshape(-1, len(self.resolution))
        if self.time_test: grid_time = time.time() - start_time

        # Compute noise and rescale
        if self.time_test: start_time = time.time()
        noise_values = np.array([self.noise(coord) for coord in coords])
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
            print("Time for computing noise and rescaling:", noise_rescaling_time, "seconds")

        return noise_values

