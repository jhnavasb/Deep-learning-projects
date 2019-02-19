# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:22:14 2018

@author: JuanMC
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel("Datos1.xlsx","Sheet5")
dfa=df.values

#print(df)
#print(dfa)
x=dfa[:,0:2]
target=dfa[:,2]
#print(x)
#print(target)
#plt.figure()
#plt.plot(x[:,0],x[:,1],'xr')

# Partiendo los datos en testing y training
from sklearn.model_selection import train_test_split
x_train, x_test, target_train, target_test= train_test_split(x,target,test_size=0.25,random_state=0)
#print(x_train)
#print(x_test)

# centrar los Datos
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)
plt.figure()

plt.plot(x[1:100,0],x[1:100,1],'xb',x[301:400,0],x[301:400,1],'xb',x[101:200,0],x[101:200,1],'xr',x[201:300,0],x[201:300,1],'xr')

import keras
from keras.models import Sequential
from keras.layers import Dense

# Crear La Rede Neuronal como un Objeto
classifier=Sequential()

classifier.add(Dense(output_dim=3, init='uniform',activation='relu',input_dim=2 ))
#classifier.add(Dense(output_dim=2, init='uniform',activation='relu' ))
classifier.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))

# Compilar la Red
classifier.compile('adam',  loss='binary_crossentropy', metrics=['accuracy'] )

#classifier.compile('rmsprop',  loss='binary_crossentropy', metrics=['accuracy'] )

#entrenamiento
classifier.fit(X_Train,target_train, batch_size=20, nb_epoch=100)

y=classifier.predict(X_Test)
y_final=np.uint8((y>0.5))
print(target_test)
print(y_final)
print(y)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test,y_final)
print(cm)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classificador():
    classificador = Sequential()
    classificador.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    classificador.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = build_classificador, batch_size = 40, epochs = 200)
efectividad = cross_val_score(estimator = classificador, X = X_Train, y = target_train, cv = 5)
promedio = efectividad.mean()
varianza = efectividad.std()
print (efectividad)
print(promedio)
print(varianza)
