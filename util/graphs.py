import matplotlib.pyplot as plt
import numpy as np


def plotTrainValAccuracy(embedding, model, acc, val_acc):
    # plot accuracy curves
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(embedding + " Embedding with " + model + " Classification Model")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plotModelComparison(embedding, cnn_acc, lstm_acc, combo_acc):
    plt.plot(cnn_acc, label='CNN Accuracy')
    plt.plot(lstm_acc, label='LSTM Accuracy')
    plt.plot(combo_acc, label='CNN & LSTM Accuracy')
    plt.title(embedding + " Embedding with Various Classification Models")
    plt.xlabel('Duration (s)')
    plt.ylabel('Epoch')
    plt.legend()
    plt.show()


def average_accuracy_for_all_embeddings(glove_cnn_acc, glove_lstm_acc, glove_combo_acc, word2vec_cnn_acc,
                                        word2vec_lstm_acc, word2vec_combo_acc):
    cnn_acc = [glove_cnn_acc, word2vec_cnn_acc]
    lstm_acc = [glove_lstm_acc, word2vec_lstm_acc]
    combo_acc = [glove_combo_acc, word2vec_combo_acc]
    cnn_acc = [(x + y) / 2 for x, y in zip(*cnn_acc)]
    lstm_acc = [(x + y) / 2 for x, y in zip(*lstm_acc)]
    combo_acc = [(x + y) / 2 for x, y in zip(*combo_acc)]

    plt.plot(cnn_acc, label='CNN Accuracy')
    plt.plot(lstm_acc, label='LSTM Accuracy')
    plt.plot(combo_acc, label='CNN & LSTM Accuracy')
    plt.title("Average accuracy for models across all embeddings")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def average_accuracy_for_all_models(glove_cnn_acc, glove_lstm_acc, glove_combo_acc, word2vec_cnn_acc,
                                    word2vec_lstm_acc, word2vec_combo_acc):
    glove_acc = [glove_cnn_acc, glove_lstm_acc, glove_combo_acc]
    word2vec_acc = [word2vec_cnn_acc, word2vec_lstm_acc, word2vec_combo_acc]
    glove_acc = [(x + y + z) / 3 for x, y, z in zip(*glove_acc)]
    word2vec_acc = [(x + y + z) / 3 for x, y, z in zip(*word2vec_acc)]
    plt.plot(glove_acc, label='GloVe Accuracy')
    plt.plot(word2vec_acc, label='Word2Vec Accuracy')
    plt.title("Average accuracy for embeddings across all models")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plotEmbeddingComparison(model, glove_acc, word2vec_acc):
    plt.plot(glove_acc, label='GloVe Accuracy')
    plt.plot(word2vec_acc, label='Word2Vec Accuracy')
    plt.title("Various Embeddings with " + model + " Classification Models")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plotTrainValLoss(embedding, model, loss, val_loss):
    # plot accuracy curves
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(embedding + " Embedding with " + model + " Classification Model")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
