# ND_path_finder
Explore N-dimensional data by traversing through the space to determine the pathways through these high-dimensional spaces while avoiding bad/negative regions. Particularly useful for control optimization, this technique allows the visualization and traversal of pathways through high-dimensional structures that are difficult to image and understand.



# Solving the problem
This project uses N-dimensional Perlin noise (typically for terrain map generation in videogames) to mimic the intricate pathways in physical systems. To simplify the data processing, we round the Perlin noise to 0 or 1 creating "good" or "bad" points respectively.

<!-- ![Perlin Noise rounded|200](https://github.com/joeyschmidt97/ND_north_star/blob/main/images/perlin_noise.png?raw=true) -->
<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/perlin_noise.png" width="512">

To create more complex data, we can increase the octaves of the Perlin noise in order to optimize performance across more complex data pathways (Left - Octave 1, Right - Octave 8).

<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/perlin_noise_octave_1.png" width="350"><img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/perlin_noise_octave_8.png" width="350">

__However__, because each of these points can be thought of as an experiment, they may be costly or timely (or both) to obtain. Therefore, we strive to obtain the boundary given minimal amount of points and construct an optimal travel pathway.

<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/perlin_noise_boundaries.png" width="350"><img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/perlin_noise_boundaries_sparse.png" width="350">


# Dataset
Our data consist of sparse 2D Perlin noise of resolution (100,100) ranging across octaves 1-9. The number of points dropped in each image ranges from 30-90% to test for model robustness across different sampling rates.



# Models

## Non-linear SVM
We used scikit's SVM with a non-linear radial basis function (RBF) kernel to fit to our datas unique curvature adapting to each data image by scanning through several values of $\gamma$ for the strength of the RBF transform and $\alpha$ for L2 (Lasso) regularization.

### Example edge reconstruction - SVM Boundary Detector (400 samples - 4% of original image)
<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/10000_2D_perlin_noise.png" width="400"><img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/10000_400sample_boundary_2D_perlin_noise.png" width="410">

### SVM boundary performance
We intuitively know that higher sampling allows the SVM boundary reconstruction to work better. We also note that more complex images (i.e. higher octaves) are more difficult to find the boundaries as well.
<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/model_performance.png" width="600">



## Voronoi Pathfinder
In order to find the optimal pathway between the boundary points we employ voronoi diagrams to tesselate the space with polygons whos edges represent the midpoints between adjacent points. 
<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/voronoi_tesselation.png" width="400">

However, we must filter only those points that are traversible (and not inside the boundary). We therefore use the gradient vectors normal to the boundary and determine the flux at each vertex of a voronoi polygon to determine if we are in/out of the boundary. This allows a programatic way to determine the optimal travel pathways in any number of dimensions.

<img src="https://github.com/joeyschmidt97/ND_path_finder/blob/main/images/voronoi_pathway_image.png" width="800">


# Installation

To get started with this repo you'll want to add the directory above to your PYTHONPATH. This will vary for different machines (i.e. Windows/Mac) so you may need to look up on a guide how to do this.

You'll then want to create an env (preferably with Anaconda) and use the provided environment.yml to load in the appropriate packages needed for this repo. This is done with the command conda env create -f environment.yml  (note you must be in the same dir as the environment.yml file)

### Installation Summary
1. Add the directory above wherever this repo is installed to your PYTHONPATH
2. Create an env from the provided environment.yml file
3. Go through examples provided in examples directories for each project



# Example Usage
The main parts of the code are split into the SVM boundary finder and the Voronoi pathfinder code. They can be found:
1. SVM boundary finder example code located in `examples/edge_detection_tests/svm_boundary.ipynb` as a python notebook
2. Voronoi pathfinder example code located in `examples/path_finders_tests/voronoi.ipynb` as a python notebook

