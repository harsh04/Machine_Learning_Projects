# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:32:46 2017

@author: Harsh Mathur
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense( activation = 'relu', units = 128))

classifier.add(Dense( activation = "sigmoid", units = 1))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=20,
                        validation_data=test_set,
                        validation_steps=2000)


fname = "weights_cnn_cat_dog.hdf5"
#load the weights
classifier.load_weights(fname)

#save the weights
classifier.save_weights(fname, overwrite=True)



from scipy.misc import imread
img = imread('predict/d.1.jpg')
img.shape
img = np.expand_dims(img, axis=0)
img.reshape()
result = classifier.predict_classes(img, batch_size=32, verbose = 1)
if result:
    print("dog")
else:
    print("cat")
