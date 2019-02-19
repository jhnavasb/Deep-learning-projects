#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import os.path as path
from scipy import ndimage, misc

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

xS, yS = 540, 960#2007, 2178 #

img = ndimage.imread("C1.png", mode="RGB")
img_resized = misc.imresize(img, (xS, yS))

img1 = ndimage.imread("T1.png", mode="L")
img1_resized = misc.imresize(img1, (xS, yS))

k, n = 0, (xS - 15) * (yS - 15)
images = []
target = np.zeros(n)


for h in range(xS - 15):
    for w in range(yS - 15):
        crop = img_resized[h:h + 15, w:w + 15]
        crop = crop.ravel()
        target[k] = img1_resized[h + 7, w + 7]
        images.append(crop)
        k += 1

images = np.asarray(images)

images, target = images / 255, target / 255
shuffled_indices = np.random.permutation(n)

x_train = images[shuffled_indices, :]
y_train = target[shuffled_indices]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

n_layers = 4
'''
def cnn():
    MIN_NEURONS = 20
    MAX_NEURONS = 180

    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    neurons = neurons.astype(np.int32)
    model = Sequential()

    for i in range(0, n_layers):
        if i == 0:
            model.add(Dense(units = neurons[i], kernel_initializer = 'uniform', activation = 'relu', input_dim = 1323))
        else:
            model.add(Dense(units = neurons[i], kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))

    model.add(Dense(units=MAX_NEURONS, kernel_initializer='uniform', activation='relu', input_dim=1))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #model.summary()

    return model
'''

def cnn2():
    model = Sequential()
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 675))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model
'''

model = KerasClassifier(build_fn = cnn2)
parameters = {'batch_size': [100, 200],
              'epochs': [20, 50, 100, 200],
              'neurons1':[8],
              'neurons2':[14],
              'neurons3':[20],
              'neurons4':[26],
              }

grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           )

grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

'''
model = cnn2()

EPOCHS = 100
BATCH_SIZE = 100
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

LOG_DIRECTORY_ROOT = 'D:\Jhonatan\Documentos\DL\LAB3\Log'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

callbacks = [early_stopping, tensorboard]
#callbacks = [tensorboard]

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)

model.save('try1.h5')

