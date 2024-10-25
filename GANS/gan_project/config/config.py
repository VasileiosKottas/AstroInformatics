# config/config.py
import torch
# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE_GEN = 0.00005
LEARNING_RATE_DISC = 0.0001
LEARNING_RATE_EMBED = 0.00005
LEARNING_RATE_REC = 0.00005
BETAS = (0.5, 0.999)
LAMBDA_GP = 10

# Model Hyperparameters
INPUT_DIM = 11          # Number of features (including wavelength)
HIDDEN_DIM = 64         # Hidden layer dimension
LATENT_DIM = 32         # Latent vector dimension
SEQUENCE_LENGTH = 17    # Length of each sequence

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
