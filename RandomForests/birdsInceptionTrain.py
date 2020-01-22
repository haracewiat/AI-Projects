# In[1]
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import itertools


# In[2]
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
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
out_layer = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=out_layer)


# In[5]
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[6]
tensorboard = TensorBoard(log_dir='.\logs')

model.fit_generator(train_generator, steps_per_epoch=32, epochs=100, callbacks=[tensorboard], verbose=2)


# In[7]
print(model.evaluate_generator(test_generator, steps=5000))


# In[8]
# unfreeze all layers for more training
for layer in model.layers:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=32, epochs=100)


# In[9]
test_generator.reset()
print(model.evaluate_generator(test_generator, steps=5000))


# In[10]
model.save("birds-inceptionv3.model")