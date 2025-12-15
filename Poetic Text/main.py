import random
import numpy as np

# Tensorflow is Machine / Deep Learning use for create AI Model
import tensorflow as tf

# keras is High Level API for Easily Coding
# Sequential Model Layer
from tensorflow.keras.models import Sequential, load_model # pyright: ignore[reportMissingImports]

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

# -- train LSTM Model for text generation character-level
# sentences = []
# next_characters = []

# -- Sliding window
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     -- Input = SEQ_LENGTH
#     sentences.append(text[i: i+SEQ_LENGTH])
#     -- Target = next 1 char
#     next_characters.append(text[i+SEQ_LENGTH])

# -- One-Hot Encoding
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool)


# # Many-to-One Sequence Learning
# # Model learn = (input(x) = prev(40char), output(y) = next(1char))
# for i, sentence in enumerate(sentences):
#     for t, character in enumerate(sentence):
#         x[i, t, char_to_index[character]] = 1
#     y[i, char_to_index[next_characters[i]]] = 1

# -- LSTM Model
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# model.fit(x, y, batch_size=256, epochs=4)

# model.save('textgenerator.keras')

model = load_model('textgenerator.keras')


# -- Random one of class from probability distribution with temperature sampling technique
def sample(preds, temperature=1.0):
    # convert input to numpy array float64( High accuracy ) 
    preds = np.array(preds).astype('float64')
    # devide probability log with temperature
    preds = np.log(preds) / temperature
    # convert back to Probability
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # random from Probability
    probas = np.random.multinomial(1, preds, 1)
    # find index 1
    return np.argmax(probas)


# -- Create new text with RNN/LSTM with predict either 1 chars
def generate_text(length, temperature):
    # random start text with SEQ length
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        # One-Hot Encoding Input
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        # Prediction next characters
        predictions = model.predict(x, verbose=0)[0]
        # Random characters from probability
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        # Update context + sliding window for Model remember last context always
        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

print('------ 0.2 ------')
print(generate_text(300, 0.2))
print('------ 0.4 ------')
print(generate_text(300, 0.4))
print('------ 0.6 ------')
print(generate_text(300, 0.6))
print('------ 0.8 ------')
print(generate_text(300, 0.8))
print('------ 1 ------')
print(generate_text(300, 1))