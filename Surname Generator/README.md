# Armenian Surname Generator

A character-level neural network that generates Armenian surnames.

The model uses an **embedding layer + MLP** to predict the next character from a fixed-length prefix.

## Pipeline

* Load and clean Armenian surnames
* Build a character vocabulary
* Create `(prefix, next character)` training examples
* Train a character-level MLP
* Search a small hyperparameter grid
* Select the best model using validation loss
* Generate surnames from an optional prefix

## Dataset

```text
data/surnames_hy.txt
```

## Model

```text
Prefix → Embedding → MLP → Next-character probabilities
```

The model is trained with cross-entropy loss and generates surnames autoregressively, one character at a time.

## Requirements

```bash
pip install numpy torch matplotlib
```

Run the notebook:

```text
src/main.ipynb
```
