import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

import tensorflow as tf
from keras.utils import pad_sequences
from keras.layers import TextVectorization, LSTM, Dense, Embedding
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Detecting Spam Email/spam_ham_dataset.csv')

ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)

balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)
balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')

train_X, test_X, train_y, test_y = train_test_split(
    balanced_data['text'], balanced_data['label'],
    test_size=0.2, random_state=42
)

vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=100,
    standardize='lower_and_strip_punctuation'
)

vectorizer.adapt(train_X)

max_len = 100

model = Sequential([
    vectorizer,
    Embedding(
        input_dim=10000,
        output_dim=32,
        input_length=max_len
    ),
    LSTM(16),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()