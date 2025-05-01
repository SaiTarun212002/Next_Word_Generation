# Next Word Generator using LSTM and GRU

This project implements a next-word prediction model using Recurrent Neural Networks (RNNs) with **LSTM** and **GRU** architectures. 
The model is trained on **Shakespeare's *Hamlet*** to generate the next word in a sequence with a focus on natural language modeling.

---

## 🔍 Project Overview

- **Objective:** Predict the next word in a sentence using sequence modeling.
- **Dataset:** [`shakespeare-hamlet.txt`](https://www.gutenberg.org/ebooks/1524)
- **Libraries Used:** `TensorFlow`, `Keras`, `NLTK`
- **Techniques:**
  - Text preprocessing with **tokenization** and **padding** (`pad_sequences`)
  - Sequence generation from text
  - Model building using **LSTM** and **GRU**
  - Evaluation based on prediction accuracy

---

## 📊 Results

| Model | Word Accuracy |
|-------|---------------|
| LSTM  | 56%           |
| GRU   | 75%           |

> Trained for **80 epochs** on preprocessed sequences.


##  Model Architecture

- **Input Layer:** Pre-padded sequences of tokenized words
- **Embedding Layer:** Converts word indices into dense vectors
- **LSTM / GRU Layer:** Learns temporal dependencies
- **Dense Output Layer:** Softmax over vocabulary to predict the next word



##  Dataset Preparation

- Used `nltk` for tokenization
- Generated sequences of words from the dataset
- Applied `pad_sequences` from Keras to ensure uniform input size


