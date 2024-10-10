import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionProcess:
    def __init__(self, steps, beta_start, beta_end):
        self.steps = steps
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)

    def add_noise(self, data, step):
        noise = torch.randn_like(data).to(device)
        noisy_data = data * torch.sqrt(1 - self.beta[step]) + noise * torch.sqrt(self.beta[step])
        return noisy_data

    def denoise(self, model, noisy_data, step):
        pred_noise = model(noisy_data)
        return noisy_data - pred_noise


