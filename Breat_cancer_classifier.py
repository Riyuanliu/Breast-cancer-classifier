import numpy as np
import pandas as pd  # Import pandas for DataFrame functionality
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Dataset
data = load_breast_cancer()

# Organize the data into a DataFrame
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at the first few rows of the DataFrame
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

#preprocess Data

#Split Data into Training and Testing Sets:
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#building and evaluating the Model

#initialize our classifier
gnb = GaussianNB();

#Train our classifer

model = gnb.fit(train, train_labels)

#make predictions

preds = gnb.predict(test)

print(preds)

#Evaluating the model's accuracy

print(accuracy_score(test_labels,preds))