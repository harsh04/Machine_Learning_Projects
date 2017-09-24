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
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

fname = "weights_cnn_cat_dog.hdf5"

#part 1 building cnn
def create_model():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense( activation = 'relu', units = 128))
    classifier.add(Dense( activation = "sigmoid", units = 1))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
    return classifier

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

classifier = create_model()
classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=20,
                        validation_data=test_set,
                        validation_steps=2000)
#save the weights
classifier.save_weights(fname, overwrite=True)


#load the weights
classifier.load_weights(fname)

#predict single image
predict_img = image.load_img('predict/d.3.jpg', target_size=(64, 64))
predict_img = image.img_to_array(predict_img)
predict_img = np.expand_dims(predict_img, axis=0)
result = classifier.predict(predict_img)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
