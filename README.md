# Text Summarization using Encoder-Decoder with Attention Mechanism

This repository contains a Python implementation of a text summarization model using an Encoder-Decoder architecture with an attention mechanism. The model takes a long input text and generates a concise summary.

## Overview

Text summarization is a natural language processing (NLP) task where the goal is to generate a shorter version (summary) of a given piece of text while retaining its key information. This implementation uses pre-trained GloVe word embeddings for better text representation and employs a sequence-to-sequence model with attention to generate summaries.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-Learn
- [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)

You can install most of these packages using `pip`:

```shell
pip install tensorflow numpy pandas scikit-learn
