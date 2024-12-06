# ND_path_finder
Explore N-dimensional data by traversing through the space to determine the pathways through these high-dimensional spaces while avoiding bad/negative regions. Particularly useful for control optimization, this technique allows the visualization and traversal of pathways through high-dimensional structures that are difficult to image and understand.


# Solving the problem
This project uses N-dimesnsional Perlin noise (typically for terrain map generation in videogames) to mimic the intricate pathways in physical systems. To simplify the data processing, we round the Perlin noise to 0 or 1 creating "good" or "bad" points respectively.

<!-- ![Perlin Noise rounded|200](https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise.png?raw=true) -->
<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise.png" width="512">

To create more complex data, we can increase the octaves of the Perlin noise in order to optimize performance across more complex data pathways.

<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_octave_1.png" width="350"><img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_octave_8.png" width="350">

__However__, because each of these points can be thought of as an experiment, they may be costly or timely (or both) to obtain. Therefore, we strive to obtain the boundary given minimal amount of points and construct an optimal travel pathway.

<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_boundaries.png" width="350"><img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise_boundaries_sparse.png" width="350">


# Dataset
Our data consist of sparse 2D Perlin noise of resolution (100,100) ranging across octaves 1-9. The number of points dropped in each image ranges from 30-90% to test for model robustness across different sampling rates.



# Models

### Non-linear SVM
We used a typical SVM with a non-linear radial basis function (RBF) kernel to fit to our datas unique curvature adapting to each data image by scanning through several values of $\gamma$ for the strength of the RBF transform and $\alpha$ for L2 (Lasso) regularization.

### Example edge reconstruction - SVM Boundary Detector
<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/10000_2D_perlin_noise.png" width="400"><img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/10000_400sample_boundary_2D_perlin_noise.png" width="400">



# Performance


<img src="https://github.com/joeyschmidt97/ND_north_star/blob/main/images/model_performance.png" width="780">

- WCE diverges while the MSE converges: Although we lose some features in the whole picture, we still guarantee pointwise accuracy at a certain level
- Zero-filling outperforms at the top of MSE: In extreme case, zero-filling guarantee 50% accuracy when kNN is out of the threshold of good performance


# Example Usage





# Installation
