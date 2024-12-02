from sklearn.svm import SVC
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


class SVMBoundary:

    def __init__(self, xdataset: xr.Dataset, num_boundary_points: int = 100, boundary_threshold: float = 0.02, boundary_spacing: float = .05):
        self.num_boundary_points = num_boundary_points
        self.boundary_threshold = boundary_threshold

        self.xdataset = xdataset
        self.X = self.xdataset['points'].values
        self.y = self.xdataset['noise_values'].values
        # Run the original grid-based method for boundary extraction
        self.guessed_boundary_points = self.svm_boundary_from_xarray()
        print('Length of boundary points (grid-based):', len(self.guessed_boundary_points))

        self.traversed_boundaries = []
        self.evenly_space_boundaries(step_size=boundary_spacing, proximity_threshold=boundary_spacing)

    def compute_gradient(self, point, epsilon=1e-5):
        """
        Computes the gradient of the decision function at a given point using finite differences.
        """
        point = np.asarray(point)
        grad = np.zeros_like(point)

        for i in range(len(point)):
            point_forward = point.copy()
            point_backward = point.copy()
            point_forward[i] += epsilon
            point_backward[i] -= epsilon

            # Extract the scalar value from the decision function
            df_forward = self.best_svm.decision_function([point_forward])[0]
            df_backward = self.best_svm.decision_function([point_backward])[0]

            grad[i] = (df_forward - df_backward) / (2 * epsilon)

        return grad

    def project_to_boundary(self, point):
        """
        Projects a point back onto the decision boundary using gradient descent.
        """
        for _ in range(100):  # Limit iterations for stability
            grad = self.compute_gradient(point)
            point -= self.best_svm.decision_function([point]) * grad / np.linalg.norm(grad)**2
            if abs(self.best_svm.decision_function([point])) < 1e-6:
                break
        return point

    def traverse_boundary(self, start_point, direction, step_size, proximity_threshold, max_steps, bounds):
        """
        Traverses a boundary iteratively from a starting point.
        """
        boundary = [start_point]
        current_point = start_point

        for _ in range(max_steps):
            grad = self.compute_gradient(current_point)
            tangent = np.array([-grad[1], grad[0]]) if direction == "forward" else np.array([grad[1], -grad[0]])
            tangent /= np.linalg.norm(tangent)
            next_point = current_point + step_size * tangent
            next_point = self.project_to_boundary(next_point)

            if len(boundary) > 1 and np.linalg.norm(next_point - boundary[0]) < proximity_threshold:
                break
            if not (bounds[0] <= next_point[0] <= bounds[1] and bounds[2] <= next_point[1] <= bounds[3]):
                break

            boundary.append(next_point)
            current_point = next_point

        return np.array(boundary)
    


    
    def svm_boundary_from_xarray(self) -> np.array:
        """
        Fits an SVM model to the data in the xarray.Dataset and plots the decision boundary.

        Parameters:
        - dataset (xarray.Dataset): The input dataset containing 'points' and 'noise_values'.

        Returns:
        - None: Plots the SVM decision boundary and data points.
        """

        # Ensure the required variables are present
        if not {'points', 'noise_values'}.issubset(self.xdataset.variables):
            raise ValueError("Dataset must contain 'points' and 'noise_values' variables.")

        # Extract features and labels
        X = self.X
        y = self.y


        # Define the parameter grid for grid search
        param_grid = {
            'C': [0.01, 0.1, 1, 5, 10, 100],  # Regularization parameter
            'gamma': [0.001, 0.1, 0.5, 1, 5, 10, 50, 100]  # Kernel coefficient
        }

        # Perform grid search to find the best parameters
        svc = SVC(kernel='rbf')
        print("grid searching... ")
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)

        # Get the best model
        print("searching best svm")
        best_svm = grid_search.best_estimator_
        print("found best one")
        self.best_svm = best_svm
        print(f"Best parameters: {grid_search.best_params_}")

        # Generate a grid of points in the input space
        bounds = [(X[:, d].min() - 0.1, X[:, d].max() + 0.1) for d in range(X.shape[1])]
        grid_points = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(10000, X.shape[1])
        )

        # Evaluate the decision function
        decision_values = best_svm.decision_function(grid_points)

        # Find boundary points (where decision value is close to 0)
        boundary_mask = np.abs(decision_values) < self.boundary_threshold
        boundary_points = grid_points[boundary_mask]

        # Filter boundary points to ensure they are within the bounds of X
        valid_mask = np.ones(len(boundary_points), dtype=bool)
        for d in range(X.shape[1]):
            valid_mask &= (boundary_points[:, d] >= X[:, d].min()) & (boundary_points[:, d] <= X[:, d].max())
        boundary_points = boundary_points[valid_mask]

        # Select 'num_boundary_points' evenly spaced boundary points
        # if len(boundary_points) > self.num_boundary_points:
        idx = np.linspace(0, len(boundary_points) - 1, self.num_boundary_points, dtype=int)
        boundary_points = boundary_points[idx]

        print(len(boundary_points))

        return boundary_points

    def evenly_space_boundaries(self, step_size=0.1, proximity_threshold=0.1, max_steps=1000):
        """
        Uses iterative traversal to find all boundaries.
        """
        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()
        grid_resolution = 100
        boundary_threshold = self.boundary_threshold

        x_vals = np.linspace(x_min, x_max, grid_resolution)
        y_vals = np.linspace(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        decision_values = self.best_svm.decision_function(grid_points)
        candidates = grid_points[np.abs(decision_values) < boundary_threshold]

        boundaries = []

        while len(candidates) > 0:
            start_point = candidates[0]
            candidates = np.delete(candidates, 0, axis=0)

            forward_boundary = self.traverse_boundary(start_point, "forward", step_size, proximity_threshold, max_steps, (x_min, x_max, y_min, y_max))
            backward_boundary = self.traverse_boundary(start_point, "backward", step_size, proximity_threshold, max_steps, (x_min, x_max, y_min, y_max))

            full_boundary = np.vstack((backward_boundary[::-1], forward_boundary))
            boundaries.append(full_boundary)

            mask = np.min(np.linalg.norm(candidates[:, None, :] - full_boundary[None, :, :], axis=2), axis=1) > proximity_threshold
            candidates = candidates[mask]

        self.traversed_boundaries = boundaries
        print(f"Found {len(boundaries)} boundaries.")

    def plot_svm_boundary(self):

        if self.X.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")

        X = self.X
        y = self.y

        boundary_points = self.guessed_boundary_points

        # Create a structured grid for visualization
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500))

        # Evaluate the decision function on the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        decision_values = self.best_svm.decision_function(grid_points)
        Z = decision_values.reshape(xx.shape)  # Now Z matches the grid shape

        # Plot the decision boundary and data points
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=5, label="Data Points")
        plt.contour(xx, yy, Z, levels=[0], linestyles=['-'], colors='k', label="Decision Boundary")
        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='red', s=10, label="Boundary Points")
        plt.title("SVM Decision Boundary with Extracted Points")
        plt.xlim(X[:, 0].min(), X[:, 0].max())
        plt.ylim(X[:, 1].min(), X[:, 1].max())
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.legend()
        plt.show()

    def plot_traversed_boundaries(self):
        if not self.traversed_boundaries:
            print("No traversed boundaries to plot.")
            return

        plt.figure(figsize=(8, 6))

        for i in range(len(self.traversed_boundaries)):
            plt.plot(self.traversed_boundaries[i][:, 0], self.traversed_boundaries[i][:, 1], ".", label="Boundary "+str(i))
        plt.title("Even Spaced Boundaries Points")
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.legend()
        plt.show()