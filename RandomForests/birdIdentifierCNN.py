"""
    Bird Identifier using CNN (Convolutional Neural Network). 
"""


# In[1]
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import itertools


# In[2]
# all images will be converted to this size
ROWS = 256
COLS = 256
CHANNELS = 3

# In[3]
train_image_generator = ImageDataGenerator(
    horizontal_flip=True, rescale=1./255, rotation_range=45)
test_image_generator = ImageDataGenerator(
    horizontal_flip=False, rescale=1./255, rotation_range=0)

train_generator = train_image_generator.flow_from_directory(
    './dataset/train', target_size=(ROWS, COLS), class_mode='categorical')
test_generator = test_image_generator.flow_from_directory(
    './dataset/test', target_size=(ROWS, COLS), class_mode='categorical')

# In[4]
train_generator.reset()
test_generator.reset()

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(ROWS, COLS, CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adamax', metrics=['accuracy'])

model.summary()


# In[5]
tensorboard = TensorBoard(log_dir='.\logs\custom')

model.fit_generator(train_generator, steps_per_epoch=512,
                    epochs=10, callbacks=[tensorboard], verbose=2)

# In[6]
print(model.evaluate_generator(test_generator, steps=1000))
