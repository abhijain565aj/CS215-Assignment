import numpy as np
import matplotlib.pyplot as plt

#custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = np.array(data)

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        x = (x - xi) / self.bandwidth
        norm_x = np.linalg.norm(x)
        if ( norm_x <= 1 ):
            return 3/4 * (1 - norm_x**2)  
        else:
            return 0.0

    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        return np.mean([ self.epanechnikov_kernel(x=x, xi=xi)* (1 / self.bandwidth)  for xi in self.data])


#load the data from the NPZ file
data_file = np.load(file='./data/2/transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kernel = EpanechnikovKDE(bandwidth=1)

# TODO: Fit the data
kernel.fit(data=data)

# TODO: Plot the estimated density in a 3D plot
range_x = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
range_y = np.linspace(data[:, 1].min(), data[:, 1].max(), 50)

x_grid, y_grid = np.meshgrid(range_x, range_y)
grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]


density = []
for point in grid_points:
    print(f"Evaluating at :{point}")
    density.append(kernel.evaluate(x=point))

density = np.array(object=density).reshape(x_grid.shape)

#creating 3d plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, density, cmap='coolwarm', edgecolor='k')

cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Density')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')

plt.title('3D Surface Plot of Density')

plt.savefig('../images/2/transaction_distribution.png')

plt.show()