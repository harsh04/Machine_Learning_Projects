# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:35:13 2017

@author: Harsh Mathur
"""

import numpy as np
import pandas as pd

#import dataset
dataset_output = pd.read_csv('gender_submission.csv')
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

X_trian = dataset_train.iloc[:, [2, 4]].values
y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.iloc[:, [1, 3]].values
y_test = dataset_output.iloc[:, 1].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_trian[:, 1] = labelencoder_X_1.fit_transform(X_trian[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_trian = onehotencoder.fit_transform(X_trian).toarray()

labelencoder_X_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_2.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()

#fitting logistic regression to train set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_trian, y_train)

#predict result
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

result = []

result = np.column_stack((dataset_test.iloc[:, 0].values ,y_pred))
import csv

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(result)


