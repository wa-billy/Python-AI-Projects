import random
import numpy as np

# Tensorflow is Machine / Deep Learning use for create AI Model
import tensorflow as tf

# keras is High Level API for Easily Coding
# Sequential Model Layer
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]

# LSTM = Long Short-Term Memory
# Dense = Fully Connected Layer
# Activate = Thinks of Neuron ( sigmoid, relu, softmax, tanh )
from tensorflow.keras.layers import LSTM, Dense, Activation # pyright: ignore[reportMissingImports]

# RMSprop = Optimized weight of Model learn
from tensorflow.keras.optimizers import RMSprop # pyright: ignore[reportMissingImports]

filepath = tf.keras.utils.get_file('shakespear.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)


# Many-to-One Sequence Learning
# Model learn = (input(x) = prev(40char), output(y) = next(1char))
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, eporch=4)