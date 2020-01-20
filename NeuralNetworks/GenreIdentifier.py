# In[1]
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical


# In[2] (Helper function to display the MFCC values)
def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()


# In[3] (See the plot)
display_mfcc('./dataset/genres/classical/classical.00067.wav')

# In[4] (Helper function to extract the MFCC values for the neural network)


def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

# In[5] (Helper function to create labels)


def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco',
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('dataset/genres/'+genre+'/*.wav')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # Create one-hot encoding for the labels
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels


# In[6] (Generate features and labels with the function above)
features, labels = generate_features_and_labels()


# In[7]
print(np.shape(features))
print(np.shape(labels))


# In[8] (Create test and training sets)
training_split = 0.8

alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx, :], alldata[splitidx:, :]

print(np.shape(train))
print(np.shape(test))

train_input = train[:, :-10]
train_labels = train[:, -10:]

test_input = test[:, :-10]
test_labels = test[:, -10:]

print(np.shape(train_input))
print(np.shape(train_labels))


# In[9] (Built the neural network)
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))


"""
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 100)               2500100   
    _________________________________________________________________
    activation_1 (Activation)    (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1010      
    _________________________________________________________________
    activation_2 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 2,501,110
    Trainable params: 2,501,110
    Non-trainable params: 0


    The output shape of the first 100 neuron layer is definitely 100 values because
    there are 100 neurons, and the output of the dense second layer is 10 because 
    there are 10 neurons. 

    So, why are there 2.5 million parameters, or weights, in the first layer? That's 
    ecause we have 25.000 inputs and each of those inputs is going to each one of the
    100 dense neurons. So that's 2.5 million, and then plus 100, because each of those 
    neurons in the 100 has its own bias term, its own bias weight, and that needs to 
    be learned as well.
"""


# %%
