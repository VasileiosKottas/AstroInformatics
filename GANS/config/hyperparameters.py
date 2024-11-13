# config/hyperparameters.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHOTOMETRY_DIM = 17
SPECTRA_DIM = 206
LATENT_DIM = 64
TRANSFORMER_HIDDEN_DIM = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 4
EPOCHS = 10000
LEARNING_RATE = 0.001
