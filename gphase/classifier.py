'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.

In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...

'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from scipy.misc import imread, imresize


# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'D:/xiong/Desktop/data/train'
validation_data_dir = 'D:/xiong/Desktop/data/train'
nb_train_samples = 420
nb_validation_samples = 420
epochs = 1
batch_size = 20

# input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

# test_data_dir = 'D:/xiong/Desktop/test/'
# test_data = ImageDataGenerator(rescale=1. / 255)
# test_generator = test_data.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False)
# # test_data = []
# # for i in range(431):
# #     img = imread('D:/xiong/Desktop/test/' + str(i + 1) + '.png', mode='RGB')
# #     # img = imresize(img,(100,100))
# #     img = np.reshape(img,[1,100,100,3])
# #     test_data.append(img)
#
# print(model.predict_generator(test_generator, steps=441))

datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)
print(bottleneck_features_train.size)
print(bottleneck_features_train)
np.save(open('bottleneck_features_train.npy', 'w'),
        bottleneck_features_train)