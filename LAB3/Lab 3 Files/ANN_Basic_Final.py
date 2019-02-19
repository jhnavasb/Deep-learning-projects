# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:27:58 2018

@author: JuanMC
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#df=pd.read_excel("Datos1.xlsx","Sheet5")
#dfa=df.values

#print(df)
#print(dfa)
#x=dfa[:,0:2]
#target=dfa[:,2]
#print(x)
#print(target)
#plt.figure()
#plt.plot(x[:,0],x[:,1],'xr')


img = cv2.imread("C2.png")
b,g,r = cv2.split(img)
x_train = np.c_[r.ravel(), g.ravel(), b.ravel()] / 255
#x_train = np.array(img.ravel() / 255).T

img = cv2.imread("C1.png")
b,g,r = cv2.split(img)
x_test = np.c_[r.ravel(), g.ravel(), b.ravel()] / 255
#x_test = np.array(img.ravel() / 255).T

target_train = (np.array([cv2.imread("T22.png", 0).ravel()])).T / 255
target_test = (np.array([cv2.imread("T11.png", 0).ravel()])).T / 255


# Partiendo los datos en testing y training
from sklearn.model_selection import train_test_split
#x_train, x_test, target_train, target_test= train_test_split(x,target,test_size=0.25,random_state=0)
#print(x_train)
#print(x_test)



# centrar los Datos
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)
#plt.figure()
#plt.plot(x[1:100,0],x[1:100,1],'xb',x[301:400,0],x[301:400,1],'xb',x[101:200,0],x[101:200,1],'xr',x[201:300,0],x[201:300,1],'xr')

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def build_clasificador(optimizer, neuronas1, neuronas2):
    clasificador = Sequential()
    clasificador.add(Dense(units = neuronas1, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3))
    clasificador.add(Dense(units = neuronas2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
    clasificador.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    clasificador.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return clasificador

clasificador = KerasClassifier(build_fn = build_clasificador)
parameters = {'batch_size': [40],
              'epochs': [10],
              'neuronas1':[2, 8],
              'neuronas2':[2, 4],
              'optimizer': ['adam']}

grid_search = GridSearchCV(estimator = clasificador,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(x_train, target_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
