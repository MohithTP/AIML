import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNNClassifier:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _majority_vote(self, labels):
        vote_count = {}
        for label in labels:
            if label in vote_count:
                vote_count[label] += 1
            else:
                vote_count[label] = 1
        
        print(vote_count)
        max_votes = -1
        prediction = None
        for label, count in vote_count.items():
            if count > max_votes:
                max_votes = count
                prediction = label
        return prediction

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = [(self._euclidean_distance(test_point, x), y) for x, y in zip(self.X_train, self.y_train)]
            distances.sort()
            k_nearest = [label for _, label in distances[:self.k]]
            predictions.append(self._majority_vote(k_nearest))
        return np.array(predictions)

# Load and split data
data = pd.read_csv('iris_csv.csv') 
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

# Train and evaluate KNN
model = KNNClassifier(k=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔍 KNN Classification Report:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("F1 Score :", f1_score(y_test, y_pred, average='macro'))