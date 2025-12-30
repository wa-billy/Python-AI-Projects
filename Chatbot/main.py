import os
import random
import json

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.relu(self.fc2(X))
        X = self.dropout(X)
        X = self.fc3(X)

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_response = []

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words