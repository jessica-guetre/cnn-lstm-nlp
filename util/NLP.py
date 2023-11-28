import keras
import numpy as np
from keras import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM
import matplotlib.pyplot as plt

from util.constants import ENCODED_VECTOR_SIZE, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, VALIDATION_SPLIT, LSTM_SIZE, GLOVE, \
    WORD2VEC, MODEL_TYPE_CNN, MODEL_TYPE_LSTM, MODEL_TYPE_CNN_LSTM


class NLPModel:
    def __init__(self, word_embedding, model_type, tokenizer, embeddings):
        self.name = f'{word_embedding} and {model_type}'
        self.word_embedding = word_embedding
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.num_words = len(tokenizer.word_index) + 1
        self.model = Sequential()
        self.misses = None
        self.hits = None
        self.embedding_matrix = None
        self.model_history = None
        if word_embedding == GLOVE:
            self.initialize_glove(embeddings)
        elif word_embedding == WORD2VEC:
            self.initialize_word2vec(embeddings)

        if model_type == MODEL_TYPE_CNN:
            self.initialize_CNN_model()
        elif model_type == MODEL_TYPE_LSTM:
            self.initialize_LSTM_model()
        elif model_type == MODEL_TYPE_CNN_LSTM:
            self.initialize_CNN_with_LSTM_model()

    def initialize_glove(self, embeddings_index):
        # Embedding Matrix
        self.hits = 0
        self.misses = 0
        # Prepare embedding matrix
        self.embedding_matrix = np.zeros((self.num_words, ENCODED_VECTOR_SIZE))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in pre-trained embeddings will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                self.hits += 1
            else:
                self.misses += 1

    def initialize_word2vec(self, embeddings):
        # Create embedding matrix
        self.embedding_matrix = np.zeros((self.num_words, ENCODED_VECTOR_SIZE))
        for word, index in self.tokenizer.word_index.items():
            if word in embeddings:
                self.embedding_matrix[index] = embeddings[word]

    def initialize_CNN_model(self):
        self.model.add(Embedding(self.num_words, ENCODED_VECTOR_SIZE,
                                 embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                                 trainable=False))
        self.model.add(Conv1D(BATCH_SIZE, 5, activation="relu"))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(BATCH_SIZE, 5, activation="relu"))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(NUM_CLASSES, activation='softmax'))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def initialize_LSTM_model(self):
        # Define LSTM model
        self.model.add(Embedding(self.num_words, ENCODED_VECTOR_SIZE,
                                 embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                                 trainable=False))
        self.model.add(LSTM(LSTM_SIZE))
        self.model.add(Dense(BATCH_SIZE, activation='softmax'))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def initialize_CNN_with_LSTM_model(self):
        # Define LSTM model

        self.model.add(
            Embedding(self.num_words, ENCODED_VECTOR_SIZE, embeddings_initializer=keras.initializers.Constant(
                self.embedding_matrix), trainable=False))
        self.model.add(Conv1D(BATCH_SIZE, 5, activation="relu"))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(BATCH_SIZE, 5, activation="relu"))
        self.model.add(LSTM(LSTM_SIZE))  # Parameter to change
        self.model.add(Dense(NUM_CLASSES, activation='softmax'))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train(self, x_train, y_train):
        self.model_history = self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                                            validation_split=VALIDATION_SPLIT)

    def plot_loss(self):
        # Plot loss curves
        plt.plot(self.model_history.history['loss'], label='Training Loss')
        plt.plot(self.model_history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for {self.name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        # Plot accuracy curves
        plt.plot(self.model_history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.model_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy for {self.name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

# %%
