# NeuralNetsNLP
20NewsGroup text classification models project for Queen's CMPE 452.

## Overview
This repository houses a suite of Jupyter notebooks and Python scripts designed for the text classification of the 20NewsGroup dataset. The project is structured to test and compare different neural network architectures and embeddings.

## Dataset
The 20NewsGroup dataset utilized for model training and evaluation is accessible via [this link](http://qwone.com/~jason/20Newsgroups/).

## Models
Implemented models in this repository include:
- **CNN (Convolutional Neural Network)**: Utilizes convolutional layers for feature extraction from text data.
- **LSTM (Long Short-Term Memory Network)**: Applies LSTM layers to capture long-term dependencies in text sequences.
- **CNN-LSTM Hybrid**: Combines CNN layers for local feature extraction with LSTM layers for capturing long-term relationships.
- **Baseline models with GloVe and Word2Vec embeddings**: To compare the performance impact of pre-trained embeddings on neural network-based classifiers.

These models are constructed to evaluate the interaction between different types of neural network layers and pre-trained embeddings, aiming to find the most effective combination for text classification.

In addition to these models, the repository includes the following:
- `constants.py`: Defines constants such as batch size, encoded vector size, and model parameters used across the models.
- `preprocessing.py`: Implements text preprocessing steps including data cleanup, stop words removal, tokenization, padding, and vectorization.
- `embeddings.py`: Handles the loading of GloVe and Word2Vec embeddings to enrich text representation.
- `graphs.py`: Provides functionality for visualizing training and validation accuracy curves to assess model performance.
- `NLP.py`: Centralizes neural network layer imports and the `NLPModel` class, encapsulating the model architecture and training process.

## Contributions
Contributors to this project include:
- Jessica Guetre - [jessica-guetre](https://github.com/jessica-guetre)
- Devin Garrow - [devgarrow](https://github.com/devgarrow)
- Tom Hamilton - [t-hamilton20](https://github.com/t-hamilton20)
- Andy Craig - [AndyCraig200](https://github.com/AndyCraig200)

## Usage
To run the notebooks, Jupyter Notebook or JupyterLab are required.
```bash
git clone https://github.com/devgarrow/NeuralNetsNLP.git
cd NeuralNetsNLP
pip install -r requirements.txt
jupyter notebook
