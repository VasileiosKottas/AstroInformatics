import torch

def create_noisy_data(data):
    noise = torch.randn_like(torch.tensor(data, dtype=torch.float32)) * 0.1
    noisy_data = data + noise
    return noisy_data, noise
