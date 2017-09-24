#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:19:54 2017

@author: harsh
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::',  header = None, engine = 'python', encoding = 'latin-1' )
users = pd.read_csv('ml-1m/users.dat', sep = '::',  header = None, engine = 'python', encoding = 'latin-1' )
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::',  header = None, engine = 'python', encoding = 'latin-1' )

#preparing test set and training set
training_set =  pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set =  pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#getting the number of user and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies =  int(max(max(training_set[:,1]), max(test_set[:,1])))

def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)