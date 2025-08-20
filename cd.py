import numpy as np
import matplotlib.pyplot as plt

def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))          # features (x, y)
    y = np.zeros(points * classes, dtype='uint8')  # class labels
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)   # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points)
        t += np.random.randn(points) * 0.2  # add noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

# Run the generator
np.random.seed(0)
X, y = spiral_data(100, 3)

# Plot: plain
plt.scatter(X[:, 0], X[:, 1])
plt.title("Spiral dataset (no class colors)")
plt.show()

# Plot: colored by class
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.title("Spiral dataset (class-colored)")
plt.show()
