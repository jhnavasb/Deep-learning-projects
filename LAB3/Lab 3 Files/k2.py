#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import os.path as path
from scipy import ndimage, misc

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

xS, yS = 900, 900#2007, 2178 #

img = ndimage.imread("SC.png", mode="RGB")
img_resized = misc.imresize(img, (xS, yS))

img1 = ndimage.imread("T_SC.png", mode="L")
img1_resized = misc.imresize(img1, (xS, yS))

r = 15
k, n, side = 0, (xS - r) * (yS - r), (r - 1) // 2
images = []
target = np.zeros(n)


for h in range(xS - r):
    for w in range(yS - r):
        crop = img_resized[h:h + r, w:w + r]
        target[k] = img1_resized[h + side, w + side]
        images.append(crop)
        k += 1

images = np.asarray(images)

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

images, target = images / 255, target / 255
shuffled_indices = np.random.permutation(n)

x_train = images[shuffled_indices, :, :]
y_train = target[shuffled_indices]

N_LAYERS = 2

def cnn(size, n_layers):
    MIN_NEURONS = 20
    MAX_NEURONS = 40
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    neurons = neurons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(neurons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(neurons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)

# Training hyperparamters
EPOCHS = 50
BATCH_SIZE = 100

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = 'D:\Jhonatan\Documentos\DL\LAB3\Log'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=1)#ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')


# Place the callbacks in a list
callbacks = [early_stopping, tensorboard, checkpoint]
#callbacks = [tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)


# Make a prediction on the test set
#test_predictions = model.predict(x_train)
#test_predictions = np.round(test_predictions)

# Report the accuracy
#accuracy = accuracy_score(y_train, test_predictions)
#print("Accuracy: " + str(accuracy))

model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)

model.save('try1.h5')

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
