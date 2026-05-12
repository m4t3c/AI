import numpy as np
from matplotlib import pyplot as plt

N_CLUSTERS = 5

X = np.random.uniform(0, 1, size=(500, 2))

plt.scatter(X[:, 0], X[:, 1]); plt.show()

C = np.random.uniform(0, 1, size=(N_CLUSTERS, 2))

def dist(x, y):
    return np.pow(x - y, 2).sum(-1).sqrt()

def k_means(dataset, centers):
    assignments = np.zeros((dataset.shape[0]),) - 1
    centers_local = centers

    while True:
        assignments_old = assignments.copy()
        assignments = np.pow(X[np.newaxis] - centers_local[:, np.newaxis], 2).sum(-1).argmin(axis=0)
        
        plt.figure(figsize=(16,10))
        plt.scatter(dataset[:, 0], dataset[:, 1])
        plt.scatter(centers_local[:, 0], centers_local[:, 1])
        plt.show()

        if np.all(assignments == assignments_old):
            break

        for i in range(len(centers_local)):
            centers_local[i] = dataset[assignments == i].mean(axis = 0)

k_means(X, C)