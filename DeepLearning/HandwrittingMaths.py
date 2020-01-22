"""
    A deep learning algorithm to recognize handwritten mathematical
    symbols.
"""

# In[1]
import keras.models
import keras.callbacks
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from IPython.display import Image
import csv
from PIL import Image as pil_image
import keras.preprocessing.image
import time

# In[2] (Display an example image from the dataset)
Image(filename="./dataset/HASYv2/hasy-data/v2-00001.png")


# In[3] (load all images (as numpy arrays) and save their classes)
imgs = []
classes = []
with open('./dataset/HASYv2/hasy-data-labels.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    i = 0
    for row in csvreader:
        if i > 0:
            img = keras.preprocessing.image.img_to_array(
                pil_image.open("./dataset/HASYv2/" + row[0]))
            # neuron activation functions behave best when input values are between 0.0 and 1.0 (or -1.0 and 1.0),
            # so we rescale each pixel value to be in the range 0.0 to 1.0 instead of 0-255
            img /= 255.0
            imgs.append((row[0], row[2], img))
            classes.append(row[2])
        i += 1


# In[4]
imgs[0]


# In[5] (Shuffle and split the data)
random.shuffle(imgs)
split_idx = int(0.8*len(imgs))
train = imgs[:split_idx]
test = imgs[split_idx:]


# In[6] (Create a matrix)

train_input = np.asarray(list(map(lambda row: row[2], train)))
test_input = np.asarray(list(map(lambda row: row[2], test)))

train_output = np.asarray(list(map(lambda row: row[1], train)))
test_output = np.asarray(list(map(lambda row: row[1], test)))


# In[7] (Convert class names into one-hot encoding)
# first, convert class names into integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(classes)

# then convert integers into one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)

# convert train and test output to one-hot
train_output_int = label_encoder.transform(train_output)
train_output = onehot_encoder.transform(
    train_output_int.reshape(len(train_output_int), 1))
test_output_int = label_encoder.transform(test_output)
test_output = onehot_encoder.transform(
    test_output_int.reshape(len(test_output_int), 1))

num_classes = len(label_encoder.classes_)
print("Number of classes: %d" % num_classes)


# In[8]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# In[9] (Visualise the performance's accuracy and valdation's accuracy with TensorBoard)
tensorboard = keras.callbacks.TensorBoard(log_dir='.\logs')


# In[10]
model.fit(train_input, train_output,
          batch_size=32,
          epochs=10,
          verbose=2,
          validation_split=0.2,
          callbacks=[tensorboard]
          )


# In[11] (Test the networks accuracy)
score = model.evaluate(test_input, test_output, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[12] (Try various model configurations and parameters to find the best)
results = []
for conv2d_count in [1, 2]:
    for dense_size in [128, 256, 512, 1024, 2048]:
        for dropout in [0.0, 0.25, 0.50, 0.75]:
            model = Sequential()
            for i in range(conv2d_count):
                if i == 0:
                    model.add(Conv2D(32, kernel_size=(
                        3, 3), activation='relu', input_shape=np.shape(train_input[0])))
                else:
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(dense_size, activation='tanh'))
            if dropout > 0.0:
                model.add(Dropout(dropout))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])

            log_dir = '.\logs\conv2d_%d-dense_%d-dropout_%.2f' % (
                conv2d_count, dense_size, dropout)
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

            start = time.time()
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard])
            score = model.evaluate(test_input, test_output, verbose=2)
            end = time.time()
            elapsed = end - start
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" %
                  (conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            results.append((conv2d_count, dense_size, dropout,
                            score[0], score[1], elapsed))


# In[13] (rebuild/retrain a model with the best parameters (from the search) and use all data)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=np.shape(train_input[0])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
# join train and test data so we train the network on all data we have available to us
model.fit(np.concatenate((train_input, test_input)),
          np.concatenate((train_output, test_output)),
          batch_size=32, epochs=10, verbose=2)

# save the trained model
model.save("mathsymbols.model")

# save label encoder (to reverse one-hot encoding)
np.save('classes.npy', label_encoder.classes_)


# In[14] (load the pre-trained model and predict the math symbol for an arbitrary image)
model2 = keras.models.load_model("mathsymbols.model")
print(model2.summary())

# restore the class name to integer encoder
label_encoder2 = LabelEncoder()
label_encoder2.classes_ = np.load('classes.npy')


def predict(img_path):
    newimg = keras.preprocessing.image.img_to_array(pil_image.open(img_path))
    newimg /= 255.0

    # do the prediction
    prediction = model2.predict(newimg.reshape(1, 32, 32, 3))

    # figure out which output neuron had the highest score, and reverse the one-hot encoding
    inverted = label_encoder2.inverse_transform(
        [np.argmax(prediction)])  # argmax finds highest-scoring output
    print("Prediction: %s, confidence: %.2f" %
          (inverted[0], np.max(prediction)))


# In[15] (predict A)
predict("./dataset/HASYv2/hasy-data/v2-00010.png")


# In[16] (predict pi)
predict("./dataset/HASYv2/hasy-data/v2-00500.png")

# In[17] (predict alpha)
predict("./dataset/HASYv2/hasy-data/v2-00700.png")
