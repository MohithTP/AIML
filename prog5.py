import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,confusion_matrix
import pandas as pd

class LogisticRegression:
    def _init_(self, learning_rate=0.05, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.theta = None

    def add_intercept(self, X):
        intercept_column = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate((intercept_column, X),axis=1)
        return X_with_intercept

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = self.add_intercept(X)
        self.theta = np.zeros((X.shape[1]))
        for _ in range(self.iteration):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= gradient * self.learning_rate

    def predict_prob(self, X):
        X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

data = pd.read_csv("Breastcancer_data.csv")

X = data.iloc[:,2:-1].values
X = np.float64(X)
y = data.iloc[:,1].values
y = np.where(y == 'M', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

model = LogisticRegression()
model.fit(X_train,y_train)

val_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, val_predictions)
precision = precision_score(y_test, val_predictions)
recall = recall_score(y_test, val_predictions)
f1 = f1_score(y_test, val_predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

confusion = confusion_matrix(y_test, val_predictions)
print(confusion)