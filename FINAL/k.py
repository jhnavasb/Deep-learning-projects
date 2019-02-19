#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
import re
import glob
import numpy as np
import os.path as path
from scipy import ndimage, misc

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import print_summary, to_categorical
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

import pickle

IMAGE_PATH = 'D:\Jhonatan\Documentos\DL\FINAL\DataManos'
#file_paths = glob.glob(path.join(IMAGE_PATH, '*.png', '*.jpg'))

file_paths = []
for ext in ('*.jpg', '*.png'):
    file_paths.extend(glob.glob(path.join(IMAGE_PATH, ext)))
images = []

for root, dirnames, filenames in os.walk(IMAGE_PATH):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (120, 160))
            images.append(image_resized)
            #plt.imshow(image_resized)
            #plt.show()
images = np.asarray(images)

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

# Scale
images = images / 255

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    labels[i] = int(filename[0])

# Split into test and training sets
TRAIN_TEST_SPLIT = 0.9

# Split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# Split the images and the labels
x_train = images[train_indices, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :]
y_test = labels[test_indices]

y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

# Hyperparamater
N_LAYERS = 4

def cnn(size, n_layers):
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (9, 9)

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
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)

'''
def cnn(size, n_layers):
    KERNEL = (3, 3)
    neurons = [32, 32, 64, 64, 128, 128]
    drop = [0.2, 0.3, 0.4]
    weight_decay = 1e-6
    k = 0
    model = Sequential()

    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(neurons[i], KERNEL, padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=shape))
        else:
            model.add(Conv2D(neurons[i], KERNEL, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if (i + 1) % 2 == 0:
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(drop[k]))
            k += 1

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dense(MAX_NEURONS))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.3))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    opt_rms = optimizers.rmsprop(lr=0.0001, decay=1e-9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_rms,
                  metrics=['accuracy'])
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)
'''
# Training hyperparamters
EPOCHS = 5
BATCH_SIZE = 32

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = 'D:\Jhonatan\Documentos\DL\LAB3\Log'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]
#callbacks = [tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)

# Report the accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])