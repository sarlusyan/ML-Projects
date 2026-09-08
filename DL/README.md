# Deep Neural Networks with Keras — MNIST Digit Classification & Transfer Learning

A hands-on exploration of training, tuning, and transferring deep neural networks in Keras/TensorFlow, built around the MNIST digits dataset. This notebook works through the core techniques for training deeper networks reliably: careful initialization, early stopping, hyperparameter search, functional-API architectures, and transfer learning between related tasks.

## What's inside

The notebook is organized into two main parts:

**1. Training and tuning a deep classifier (digits 0–4)**
- A 5-hidden-layer fully connected network (`ELU` activations, He initialization) trained to classify MNIST digits 0–4
- Training stabilized with `EarlyStopping` and `ModelCheckpoint` callbacks
- A parameterized `build_model()` function wrapped in `scikeras.KerasClassifier` for use with scikit-learn tooling
- Hyperparameter search over network depth, width, and learning rate using `GridSearchCV`

**2. Transfer learning experiment (digits 5–9)**
- Two identical 5-layer "twin" branches combined via the Keras functional API into a pairwise comparison model (siamese-style architecture), trained on digit pairs to learn a general similarity/feature representation
- One trained branch is then frozen (`trainable = False`) and reused as a fixed feature extractor
- A new classifier head is trained on top of the frozen branch to classify a *different* set of digits, demonstrating how learned features transfer to a related but distinct task

## Results

| Stage | Task | Result |
|---|---|---|
| Base classifier | Classify digits 0–4 | **99.14%** test accuracy |
| Pairwise/twin network | Learn shared digit features | 93.6% batch accuracy after 5 epochs |
| Transfer-learned classifier | Classify digits (frozen feature extractor + new head) | **96.8%** validation accuracy |

The transfer learning result shows the frozen branch — trained only on a pairwise similarity task — captured features general enough to support strong accuracy on a new classification task with just a lightweight new head.

## Tech stack

- **TensorFlow / Keras** — model building (Sequential & Functional APIs), training, callbacks
- **scikit-learn** — `GridSearchCV` for hyperparameter tuning
- **scikeras** — bridges Keras models into scikit-learn's estimator interface
- **NumPy** — data preprocessing

## Key techniques demonstrated

- He initialization + ELU activations for training deeper networks
- Early stopping and model checkpointing to avoid overfitting
- Systematic hyperparameter search (depth, width, learning rate) via cross-validation
- The Keras Functional API for multi-input, multi-branch architectures
- Layer/branch freezing for transfer learning
- Manual batch-pair generation and `train_on_batch` for custom training loops
