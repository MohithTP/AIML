import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeansClustering:
    def _init_(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        self.X = np.array(X)
        n_samples, _ = self.X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = self.X[random_indices]

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(self.X, self.centroids)
            new_centroids = np.array([self.X[self.labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X, centroids):
        labels = []
        for point in X:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)
    
    def plot_custom_clusters(self, dim1=2, dim2=3):
        colors = ['blue', 'green', 'orange','purple','pink']
        for i in range(self.k):
            cluster_points = self.X[self.labels == i]
            plt.scatter(cluster_points[:, dim1], cluster_points[:, dim2], label=f'Cluster {i}', color=colors[i], alpha=0.6)
        
        # Plot centroids
        plt.scatter(self.centroids[:, dim1], self.centroids[:, dim2], marker='X', s=50, c='red', label='Centroids')

        # Labels and title
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.title('Custom K-means Clustering on Iris Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()

# Load dataset
data = pd.read_csv('iris_csv.csv') 
x = data.iloc[:, :-1].values  # Only features

# Train model
model = KMeansClustering(k=4)
model.fit(x)

model.plot_custom_clusters(dim1=0, dim2=1)