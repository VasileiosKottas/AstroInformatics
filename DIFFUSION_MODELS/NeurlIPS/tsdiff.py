import torch
import torch.nn as nn
from residual_block import ResidualBlock

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
        self.initial_conv = nn.Conv1d(in_channels=17, out_channels=32, kernel_size=3, padding=1)

        # Linear layer to match d_model
        self.to_d_model = nn.Conv1d(32, d_model, kernel_size=1)

        # Stack of Residual Blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(d_model, d_model) for _ in range(num_blocks)
        ])

        # Transformer Encoder Layer
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_encoder_layers
        )

        # Output Convolution Layer
        self.output_conv = nn.Conv1d(in_channels=d_model, out_channels=206, kernel_size=1)

    def forward_diffusion_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).unsqueeze(-1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def forward(self, x, t):
        x = self.initial_conv(x)  # Shape: [batch_size, 32, seq_len]

        x = self.to_d_model(x)  # Shape: [batch_size, d_model, seq_len]

        for block in self.residual_blocks:
            x = block(x)

        # Prepare for Transformer
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, d_model]
        x = self.transformer_layer(x)  # Shape: [seq_len, batch_size, d_model]
        x = x.permute(1, 2, 0)  # Back to [batch_size, d_model, seq_len]

        # Final output convolution
        x = self.output_conv(x)  # Shape: [batch_size, 206, seq_len]
        return x


    def denoise(self, noisy_x, t):
        # Ensure the tensor has the correct shape
        # if noisy_x.size(1) != 17:  # Check if channels are not 17
        #     noisy_x = noisy_x.permute(0, 2, 1)  # Permute to [batch_size, in_channels, seq_len]
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

    def sample(self, num_samples, device, sequence_length=1):
        noise = torch.randn((num_samples, 17, sequence_length)).to(device)

        for t in reversed(range(self.ts_diff.time_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise = self.ts_diff.denoise(noise, t_tensor)
        
        return noise
