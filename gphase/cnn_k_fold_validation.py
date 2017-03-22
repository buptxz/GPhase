'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

batch_size = 30
num_classes = 2
epochs = 10

# input image dimensions
img_rows, img_cols = 100, 100

x_data = np.load("../data/x_data.npy")
y_data = np.load("../data/y_data.npy")
y_data_original = np.load("../data/y_data.npy")

# the data, shuffled and split between train and test sets
if K.image_data_format() == 'channels_first':
    x_data = x_data.reshape(x_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_data = x_data.astype('float32')
x_data /= 255
print('x_data shape:', x_data.shape)
print(x_data.shape[0], 'data samples')

# convert class vectors to binary class matrices
y_data = keras.utils.to_categorical(y_data, num_classes)

# define 10-fold cross validation test harness
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_scores = []

for train, test in k_fold.split(x_data, y_data_original):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_data[train], y_data[train], batch_size=batch_size, epochs=epochs, verbose=1)
    score = model.evaluate(x_data[test], y_data[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    cv_scores.append(score[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))