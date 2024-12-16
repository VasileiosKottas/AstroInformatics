import torch
import torch.nn as nn
from .residual_block import ResidualBlock

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
        self.to_d_model = nn.Linear(32, d_model)

        # Stack of Residual Blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(32, 32) for _ in range(num_blocks)
        ])

        # Transformer Layer
        self.transformer_layer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)

        # Conv Layer
        self.output_conv = nn.Conv1d(in_channels=d_model, out_channels=206, kernel_size=1)

    def forward_diffusion_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def forward(self, x, t):
        print(f"Forward input shape: {x.shape}")
        x = self.initial_conv(x)
        print(f"Shape after initial_conv: {x.shape}")
        
        for block in self.residual_blocks:
            x = block(x)
        print(f"Shape after residual blocks: {x.shape}")
        
        # Transformer
        x = self.to_d_model(x.permute(0, 2, 1))  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer_layer(x, x)
        x = x.permute(1, 2, 0)  # [batch, d_model, seq_len]
        print(f"Shape after transformer: {x.shape}")

        # Output
        output = self.output_conv(x)
        print(f"Shape after output_conv: {output.shape}")
        return output



    def denoise(self, noisy_x, t):
        print(f"Denoise input shape: {noisy_x.shape}")  # Debugging
        predicted_noise = self.forward(noisy_x, t)
        
        # Ensure the output shape matches the model's expected input shape
        if predicted_noise.size(1) != 17:  # If channels are not 17, reshape
            predicted_noise = predicted_noise[:, :17, :]  # Slice to match required shape

        print(f"Denoise output shape: {predicted_noise.shape}")  # Debugging
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
        print("Denoise in Self",noisy_x.shape)
        return self.ts_diff.denoise(noisy_x, t)
    
    def sample(self, num_samples, device, sequence_length=1):
        # Initialize noise with the correct number of input channels (17)
        noise = torch.randn((17, 206)).to(device)
        
        # Perform reverse diffusion through the model
        for t in reversed(range(self.ts_diff.time_steps)):
            
            
            noise = self.ts_diff.denoise(noise, t)
        
        return noise
