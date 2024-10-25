import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Residual Block as depicted in the diagram
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.swish(self.conv1(x))
        out = self.conv2(out)
        out += x  # Residual connection
        return self.relu(out)

# TSDiff Architecture based on the diagram
class TSDiff(nn.Module):
    def __init__(self, input_dim, time_steps=1000, num_blocks=5, d_model=64, nhead=4, num_encoder_layers=2):
        super(TSDiff, self).__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.d_model = d_model

        # Noise schedule (beta values)
        self.betas = torch.linspace(0.0001, 0.02, time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        # Initial Convolution Layer
        self.initial_conv = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)

        # Linear layer to match d_model
        self.to_d_model = nn.Linear(32, d_model)

        # Stack of Residual Blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(32, 32) for _ in range(num_blocks)
        ])

        # Transformer Layer
        self.transformer_layer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)

        # Output Layer
        self.output_conv = nn.Conv1d(in_channels=d_model, out_channels=input_dim, kernel_size=3, padding=1)

    def forward_diffusion_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def forward(self, x, t):
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)

        # Project to d_model dimensions
        x = self.to_d_model(x.permute(0, 2, 1))  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]

        # Pass through the Transformer layer
        x = self.transformer_layer(x, x)
        x = x.permute(1, 2, 0)  # [batch, d_model, seq_len]

        output = self.output_conv(x)
        return output

    def denoise(self, noisy_x, t):
        predicted_noise = self.forward(noisy_x, t)
        return predicted_noise


class SelfGuidedTSDiff(nn.Module):
    def __init__(self, input_dim, time_steps=1000):
        super(SelfGuidedTSDiff, self).__init__()
        self.ts_diff = TSDiff(input_dim=input_dim, time_steps=time_steps)

    def forward(self, xt, t, y_obs=None):
        eps_hat = self.ts_diff(xt, t)
        if y_obs is not None:
            grad = torch.autograd.grad(eps_hat, xt, grad_outputs=torch.ones_like(eps_hat), create_graph=True)[0]
            eps_hat = eps_hat + 0.1 * grad * (y_obs - eps_hat)
        return eps_hat

    def denoise(self, noisy_x, t):
        return self.ts_diff.denoise(noisy_x, t)
