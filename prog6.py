import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NaiveBayes():
    def _init_(self):
        self.class_prob={}
        self.features_prob={}
    
    def fit(self,x,Y):
        #calculate frequency of each
        for value in Y:
            if value in self.class_prob.keys():
                self.class_prob[value] += 1
            else:
                self.class_prob[value] = 1

        counts = list(self.class_prob.values())    ##counts
        total_samples = len(Y)

        #calculate probability
        for key in self.class_prob:
            self.class_prob[key]=self.class_prob[key]/total_samples

        #probabioty of feature given class
        for c in self.class_prob:
            self.features_prob[c] = {} #dict inside dict for features
            for feature in x.columns:
                self.features_prob[c][feature] = {} #one more dict inside to store
                unique_values = x[feature].unique() #unique values of that feature
                for value in unique_values:
                    count = np.sum((x[feature] == value) & (Y == c))
                    self.features_prob[c][feature][value] = count / counts[c]
        # print(self.features_prob)
                    
    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            row = x.iloc[i]
            max_prob = -1
            predicted_class = None
            for c in self.class_prob:
                prob = self.class_prob[c]
                for feature in x.columns:
                    value = row[feature]
                    if value in self.features_prob[c][feature]:
                        prob *= self.features_prob[c][feature][value]
                    else:
                        prob *= 0
                if prob > max_prob:
                    max_prob = prob
                    predicted_class = c
            predictions.append(predicted_class)
        return predictions

data = pd.read_csv("Social_Network_Ads.csv")
data["Gender"] = np.where(data["Gender"] == 'Male', 1, 0)
X = data.iloc[:,1:4]
y = data['Purchased']
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=NaiveBayes()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

accuracy = accuracy_score(Y_pred, Y_test)
precision = precision_score(Y_pred, Y_test)
recall = recall_score(Y_pred, Y_test)
f1 = f1_score(Y_pred, Y_test)

print("Validation Set Metrics:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
confusion = confusion_matrix(Y_pred, Y_test)
print(confusion)