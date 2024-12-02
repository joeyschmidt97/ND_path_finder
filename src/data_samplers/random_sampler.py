
import random
import xarray as xr

def sample_xarray(dataset, n_samples):
    """
    Randomly samples a specified number of points from an xarray.Dataset
    and returns a new xarray.Dataset with the sampled points and noise_values.
    
    Args:
    - dataset (xarray.Dataset): Original dataset containing 'points' and 'noise_values'.
    - n_samples (int): Number of points to sample.
    
    Returns:
    - xarray.Dataset: A new dataset with the sampled points and corresponding noise_values.
    """
    # Ensure we don't sample more points than available
    total_points = len(dataset['points'])
    if n_samples > total_points:
        raise ValueError(f"Requested {n_samples} samples, but only {total_points} points are available.")
    
    # Randomly sample indices
    sampled_indices = random.sample(range(total_points), n_samples)
    
    # Select sampled points and noise_values
    sampled_points = dataset['points'].values[sampled_indices]
    sampled_noise_values = dataset['noise_values'].values[sampled_indices]
    
    # Create a new xarray.Dataset with the sampled data
    sampled_dataset = xr.Dataset(
        {
            'noise_values': (['points'], sampled_noise_values),
        },
        coords={
            'points': (['points', 'dim'], sampled_points),
        },
        attrs=dataset.attrs  # Copy over the original attributes
    )
    
    return sampled_dataset
