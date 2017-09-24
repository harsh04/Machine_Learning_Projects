# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:35:03 2017

@author: Harsh Mathur
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#getting input and output
X_trian = training_set[0: 1257]
y_train = training_set[1: 1258]

#reshape
X_trian = np.reshape(X_trian, (1257, 1, 1))

#part 2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()
#input layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
#output layer
regressor.add(Dense(units = 1))
#compile rnn
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_trian, y_train, batch_size = 32, epochs = 200)

#part-3 making predictions
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values
#predicting 
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualise the result
plt.plot(real_stock_price, color = 'red', label = 'Real google stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('stock price')
plt.legend()
plt.show()









