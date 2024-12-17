# Text-Classifier using Convolutional Neural Networks (CNN)

## Overview
This project implements a sentiment analysis model using Convolutional Neural Networks (CNNs) in Python. The code processes a dataset of text reviews, applies data preprocessing and text embedding with GloVe, and builds a deep learning model to classify sentiments into positive or negative categories.


## Features
- **Text Preprocessing**: Includes cleaning, tokenization, and sequence padding.
- **Embedding Layer**: Utilizes GloVe pre-trained embeddings for better semantic understanding.
- **CNN Architecture**: Implements a simplified and a complex CNN model with multiple convolutional and pooling layers.
- **Sentiment Classification**: Outputs binary classification (positive/negative sentiment).


## Requirements
- Python 3.x
- Required Libraries: 
  - `numpy`
  - `pandas`
  - `tensorflow`
  - `bs4` (BeautifulSoup)
  - `pickle`


## Dataset
- The project uses the IMDB dataset (`labeledTrainData.tsv`), where each review is labeled with a sentiment (positive/negative).
- Pre-trained GloVe embeddings (`glove.6B.100d.txt`) are used for text representation.


## Usage
1. Clone the repository and install the required libraries.
2. Place the `labeledTrainData.tsv` and GloVe embeddings in the appropriate directories.
3. Run the script to train the CNN models:
   ```bash
   python script_name.py

## Required Datasets

1. **IMDB Dataset for Sentiment Analysis**  
   Download the IMDb train data from Kaggle:
   - [Download IMDb train data from Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/download/labeledTrainData.tsv)  
   After downloading, place the file `labeledTrainData.tsv` in your working directory.

2. **GloVe Word Vectors**  
   Download the GloVe word vectors (6B tokens, 400K vocabulary, 50/100/200/300 dimensions):
   - [Download GloVe word vectors](https://nlp.stanford.edu/data/glove.6B.zip)  
   After downloading, unzip the `glove.6B.zip` file and place the extracted files in your working directory.
