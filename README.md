# NeuralNetsNLP - 20NewsGroup Text Classification Models

## Overview
This repository contains a series of Jupyter notebooks and Python scripts implementing various deep learning models for text classification on the 20NewsGroup dataset. Each model explores different neural network architectures and embeddings to classify text data into one of 20 newsgroups.

## Dataset
The dataset can be downloaded from [this link](http://qwone.com/~jason/20Newsgroups/).

## Models
The models included in this repository are:

- `GloveCNNModel.ipynb`: A Convolutional Neural Network (CNN) model using GloVe embeddings for text classification.
- `GloveLSTMModel.ipynb`: A Long Short-Term Memory (LSTM) model using GloVe embeddings.
- `GloveLSTMwithCNNModel.ipynb`: An integrated model combining LSTM and CNN layers using GloVe embeddings.
- `word2vec_CNN.ipynb`: A CNN model using Word2Vec embeddings.
- `word2vec_CNN_pretrained.ipynb`: A CNN model with pre-trained Word2Vec embeddings.
- `word2vec_CNN_with_LSTM_pretrained.ipynb`: A combined CNN and LSTM model with pre-trained Word2Vec embeddings.
- `word2vec_LSTM.ipynb`: An LSTM model using Word2Vec embeddings.
- `word2vec_LSTM_pretrained.ipynb`: An LSTM model with pre-trained Word2Vec embeddings.
- `ModelComparison.ipynb`: A notebook for comparing the performance of the different models.

In addition to these models, the repository includes a preprocessing script:

- `preprocessing.py`: A Python script for text data preprocessing which is used across all models.

## Contributions

Contributors to this project include:

- Jessica Guetre - [jessica-guetre](https://github.com/jessica-guetre)
- Devin Garrow - [devgarrow](https://github.com/devgarrow)
- Tom Hamilton - [t-hamilton20](https://github.com/t-hamilton20)
- Andy Craig

## Usage
To run the notebooks, make sure you have Jupyter Notebook or JupyterLab installed. You can launch the notebooks using the following command:

```bash
git clone https://github.com/devgarrow/NeuralNetsNLP.git
cd NeuralNetsNLP
pip install -r requirements.txt
jupyter notebook
