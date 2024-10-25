import matplotlib.pyplot as plt
import torch

def plot_real_vs_generated(real_data, generated_data, scaler, num_samples=5):
    """
    Plots a comparison between real and generated data.
    
    Parameters:
    - real_data: torch.Tensor, the real data batch (shape: [batch_size, sequence_length, num_features])
    - generated_data: torch.Tensor, the generated data batch (shape: [batch_size, sequence_length, num_features])
    - scaler: fitted scaler for inverse normalization
    - num_samples: int, the number of samples to plot (default is 5)
    """
    
    # Move data to CPU if necessary
    real_data = real_data.detach().cpu().numpy()
    generated_data = generated_data.detach().cpu().numpy()
    
    # De-scale the generated data
    batch_size, sequence_length, num_features = generated_data.shape
    generated_data = generated_data.reshape(batch_size * sequence_length, num_features)
    generated_data = scaler.inverse_transform(generated_data)
    generated_data = generated_data.reshape(batch_size, sequence_length, num_features)
    
    # Ensure num_samples does not exceed batch size
    num_samples = min(num_samples, real_data.shape[0])
    
    # Plot each sample
    plt.figure(figsize=(10, 6 * num_samples))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        
        # Plot real data vs generated data for each feature
        plt.plot(real_data[i, :, 0], real_data[i, :, 1:], label='Real', color='blue', alpha=0.7)
        plt.plot(generated_data[i, :, 0], generated_data[i, :, 1:], label='Generated', linestyle='dashed', color='red', alpha=0.7)
        
        plt.legend()
        plt.title(f'Sample {i + 1} - Real vs Generated')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
    
    plt.tight_layout()
    plt.show()
