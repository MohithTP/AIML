import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class kmeans():
    def __init__(self, iter= 1000, k = 4, tol = 1e-4):
        self.k = k
        self.iterations = iter
        self.tol = tol
        
    def fit(self, x):
        self.X = np.array(x)
        n, _ = self.X.shape
        rand = np.random.choice(n, self.k, replace=False)
        self.centroids = self.X[rand]
        
        for _ in range(self.iterations):
            self.labels = []
            for point in self.X: 
                dist = [np.linalg.norm(point-c) for c in self.centroids]
                self.labels.append(np.argmin(dist))
            self.labels = np.array(self.labels)
            new = np.array([self.X[self.labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.linalg.norm(self.centroids - new, axis=1) < self.tol):
                break
            self.centroids = new
        
    def plot(self, dim1 = 2, dim2 = 3):
        colors = ['blue', 'green', 'red', 'orange','cyan']
        for i in range(self.k):
            points = self.X[self.labels == i]
            plt.scatter(points[:,dim1], points[:,dim2], label = f'Cluster{i}',color = colors[i], alpha=0.6)
            
        plt.scatter(self.centroids[:,dim1], self.centroids[:,dim2], marker="X", s=200,c = 'red', label = "Centroids")
        
        plt.xlabel("Petal length")
        plt.ylabel("Petal width")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
data = pd.read_csv("iris_csv.csv")

x = data.iloc[:,:-1].values

model = kmeans()
model.fit(x)
model.plot()