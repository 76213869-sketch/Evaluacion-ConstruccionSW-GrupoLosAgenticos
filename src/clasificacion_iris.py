
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility to ensure the same clusters each time
np.random.seed(42)

# --- Generate data for Class 0 (Left Cluster: compact blue circles) ---
mean_0 = [3, 4]  # Center of the left cluster
cov_0 = [[0.2, 0], [0, 0.2]] # Small, diagonal covariance for compact, circular shape
X_0 = np.random.multivariate_normal(mean_0, cov_0, 100) # 100 points for the cluster

# --- Generate data for Class 1 (Right Cluster: compact purple squares) ---
mean_1 = [7, 6]  # Center of the right cluster, symmetric to mean_0 but shifted
cov_1 = [[0.2, 0], [0, 0.2]] # Same compact, circular shape
X_1 = np.random.multivariate_normal(mean_1, cov_1, 100) # 100 points for the cluster

# --- Define the perfect diagonal decision boundary ---
# The midpoint between the two cluster means ([3,4] and [7,6]) is ([5,5]).
# The line connecting the means has slope (6-4)/(7-3) = 2/4 = 0.5.
# The perpendicular bisector (our decision boundary) will have a slope of -1/0.5 = -2.
# Using point-slope form: y - y1 = m(x - x1)
# y - 5 = -2 * (x - 5)
# y = -2x + 10 + 5
# y = -2x + 15
m = -2 # Slope of the decision boundary
c = 15 # Y-intercept of the decision boundary

# Create x values for the decision boundary line, extending across the plot range
x_boundary = np.array([0, 10])
y_boundary = m * x_boundary + c

# --- Create the plot ---
fig, ax = plt.subplots(figsize=(8, 6)) # Set figure size for clean visualization

# Plot Class 0 points (blue circles)
ax.scatter(X_0[:, 0], X_0[:, 1], color='blue', marker='o', s=60, label='Class 0', zorder=2)

# Plot Class 1 points (purple squares)
ax.scatter(X_1[:, 0], X_1[:, 1], color='purple', marker='s', s=60, label='Class 1', zorder=2)

# Plot the decision boundary
ax.plot(x_boundary, y_boundary, color='black', linestyle='-', linewidth=2.5, label='Decision Boundary', zorder=1)

# --- Customizations for minimal, academic style ---
ax.set_facecolor('white') # White background for the plot area
fig.patch.set_facecolor('white') # White background for the figure itself

ax.set_xticks([]) # Remove x-axis ticks
ax.set_yticks([]) # Remove y-axis ticks
ax.set_xlabel('') # Remove x-axis label
ax.set_ylabel('') # Remove y-axis label

ax.set_xlim(0, 10) # Set x-axis limits to frame the data well
ax.set_ylim(2, 8)  # Set y-axis limits to frame the data well

ax.set_aspect('equal', adjustable='box') # Maintain equal aspect ratio for symmetry

plt.grid(False) # Ensure no grid lines are displayed
plt.title('Perfect Binary Classification Example', fontsize=16, color='black') # Add a clean title
plt.tight_layout() # Adjust plot to ensure everything fits
plt.show()
