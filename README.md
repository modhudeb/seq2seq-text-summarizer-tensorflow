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
```

# Usage

## Data Preparation:

1. Ensure your dataset is organized with a DataFrame containing 'Text' and 'Summary' columns.
2. Data cleaning and preprocessing steps have been implemented to remove HTML tags, punctuation, and extra whitespace. Adjust them as needed.

## Model Configuration:

1. Set your desired hyperparameters such as `max_text_length`, `max_summary_length`, `latent_dim`, and the path to your GloVe embeddings (`glove_path`).

## Training:

1. Run the provided code to train the model on your data. The code includes K-Fold cross-validation for robust training. You can adjust the number of epochs and batch size as needed.

## Inference:

1. After training, you can use the model to generate summaries for new input text using the `generate_summary` function provided in the code. Simply pass the input text to the function to obtain a predicted summary.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The code uses pre-trained GloVe word embeddings, which are available at [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).
- Special thanks to the TensorFlow and scikit-learn communities for their powerful libraries.

Feel free to fork this repository, modify the code, and adapt it to your specific needs. If you encounter any issues or have questions, please open an issue in the [GitHub repository](https://github.com/modhudeb/seq2seq-text-summarizer-tensorflow).

