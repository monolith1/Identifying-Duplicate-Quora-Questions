# Identifying Duplicate Quora Questions via NLP

Identification of duplicate questions on Quora, a popular website where users seek answers to specific questions.

![Header Image](https://qph.fs.quoracdn.net/main-qimg-98bcd12eeefce664a95e35366b69af92-c)

## Problem Domain

Over 100 million people visit Quora every month, so it's no surprise that many people ask similar (or the same) questions. Various questions with the same intent can cause people to spend extra time searching for the best answer to their question, and results in members answering multiple versions of the same question.

By identifying these duplicate questions, we can significantly improve the user experience by delivering higher quality content. Rather than having to reference multiple ways of asking the same question, all of the answers will be in one place. This also works to reduce the operational overhead of having duplicate assets. 

This approach is being used here to isolate duplicate questions, however, it is easy to imagine it applied to areas such as user support or search optimization.


## Data

The labeled dataset can be downloaded from [here](https://drive.google.com/file/d/19iWVGLBi7edqybybam56bt2Zy7vpf1Xc/view?usp=sharing).

## Project Overview

In order to compare questions, several preprocessing steps were applied:
* Punctuation removal
* Conversion to lowercase
* Removal of leading and trailing spaces
* Tokenization
* Lemmatization
* Removal of stopwords and non-alphanumeric words
* TF-IDF vectorization

Several ML models were trained on the dataset:
* RandomForest (82% accuracy on test data)
* XGBoost (80% accuracy on test data)
* Multinomial Naive Bayes (75% accuracy on test data)
* Logistic Regression (74% accuracy on test data)

Additionally, two deep learning models were trained on the dataset. These models employed siamese LSTM architecture in order to capture and compare contextual information. The preprocessing for deep learning included:
* Tokenization
* Padding
* Sequencing
* Limitation of sequence length

![Siamese LSTM](https://user-images.githubusercontent.com/34228896/42486558-45490070-841a-11e8-9cf3-61cdea15de1d.png)

One model employed GloVe pre-trained word embeddings, while the other employed word2vec pre-trained word embeddings. Both achieved approximately 75% accuracy on the test data.

## Repository Overview

This repository contains the following files:

```
├───DeepLearning_GloVe.ipynb  
├───DeepLearning_w2v.ipynb
├───ML.ipynb
├───NLPDuplicate.pptx
├───preprocessed.csv
├───README.md  
```

* **DeepLearning_GloVe.ipynb** and **DeepLearning_w2v.ipynb** contain all of the modeling for the Keras/Tensorflow neural networks.
* **ML.ipynb** contains all of the modelling for Machine Learning model.
* **NLPDuplicate.pptx** contains presentation slides detailing the project.
* **preprocessed.csv** contains preprocessed data for ML model training.

