#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:11:37 2019

@author: guckert
"""

# Load libraries
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Helper function
def make2ColArray(x):
    res = []
    for row in x:
        res.append([row[0], row[1]])
        res.append([row[2], row[3]])
        res.append([row[4], row[5]])
        res.append([row[6], row[7]])

    return res


def predictAndPlot(model,point):
    angle=True
    # Prediction
    print(point)
    result = model.predict(np.array([point]), verbose=0)

    print(result)

    x1 = []
    y1 = []

    for i in range(len(result[0])):
        if i % 2 == 0:
            x1.append(result[0][i])
        else:
            y1.append(result[0][i])

    x = point[0]
    y = point[1]

    #plt.ylim(-10, 10)
    #plt.xlim(-10, 10)
    plt.plot([x], [y], 'bo')
    plt.plot([x1], [y1], 'ro')

    if angle==True:
        plt.plot([0,x],[0,y],linewidth=1,color='black')
        plt.plot([0,x1[0]],[0,y1[0]],linewidth=1,color='black')

def createModel():
    # Define neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=4, activation='relu'))

    # Add fully connected layer with no activation function
    network.add(layers.Dense(units=2))

    # Compile neural network
    network.compile(loss='mse', optimizer='RMSprop',
                    metrics=['mse'])  # Mean squared error  # Optimization algorithm  # Mean squared error

    return network

def loadModel():
    network = models.load_model('saved_model/my_model')
    return network

def trainModel():
    # Set random seed
    np.random.seed(0)

    ### Input Data
    #data = pd.DataFrame(columns=['x1','y1','x1_rotated','y1_rotated','x2','y2','x2_rotated','y2_rotated','x3','y3','x3_rotated','y3_rotated','x4','y4','x4_rotated','y4_rotated'])
    data = pd.read_csv('data.csv', encoding='utf8', header=0, sep=';')

    x_train_hlp=data[['x1','y1','x2','y2','x3','y3','x4','y4']].values
    x_train = make2ColArray(x_train_hlp)

    ### Target Data
    y_train_hlp = data[['x1_rotated','y1_rotated','x2_rotated','y2_rotated','x3_rotated','y3_rotated','x4_rotated','y4_rotated']].values
    y_train = make2ColArray(y_train_hlp)

    # Spilt data
    train_features, test_features, train_target, test_target = train_test_split(x_train, y_train, test_size=0.1)

    # Convert to numpy arrays
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_target = np.array(train_target)
    test_target = np.array(test_target)

    # Print
    print('train_data_shape: ', np.shape(train_features))
    print('train_features example: ', train_features[0])
    print('train_target example: ', train_target[0])
    print('test_data_shape: ', np.shape(test_features))

    network=createModel()

    # Train neural network
    history = network.fit(train_features, train_target, epochs=1, verbose=1, batch_size=100, validation_data=(test_features, test_target))

    # Show architecture of model
    network.summary()

    network.save('saved_model/my_model')

    return network

def doit(model,points):
    for i in points:
        predictAndPlot(model,i)

    plt.show()

def lambd(model):
    points=[[1,1],[2,2],[3,3],[4,4],[5,4],[6,3],[7,2],[8,1],[4,5],[3,6],[2,7],[1,8]]
    doit(model,points)

def rect(model):
    points=[[1,2],[1,8],[3,2],[3,8]]
    doit(model,points)

def sp(model):
    predictAndPlot(model,[1,8])

    plt.show()

def newModel():
    return trainModel()

def useModel():
    return loadModel()

def main():
    model=trainModel()
    lambd(model)



if __name__ == "__main__":
    main()
