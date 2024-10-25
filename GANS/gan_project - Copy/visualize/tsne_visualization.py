from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_tsne(real_data, generated_data, scaler, device, num_samples=32):
    """
    Visualizes the real and generated data using t-SNE.
    
    Parameters:
    - real_data: torch.Tensor, the real data batch (shape: [num_samples, sequence_length, num_features])
    - generated_data: torch.Tensor, the generated data batch (shape: [num_samples, sequence_length, num_features])
    - scaler: fitted scaler for inverse normalization
    - device: torch.device, the device being used (CPU or GPU)
    - num_samples: int, the number of samples to visualize (default is 32)
    """
    
    sequence_length = real_data.shape[1]
    num_features = real_data.shape[2]
    
    # Flatten the real and generated data to be of shape (num_samples * sequence_length, num_features)
    real_data_cpu = real_data.detach().cpu().numpy().reshape(num_samples * sequence_length, num_features)
    generated_data_cpu = generated_data.detach().cpu().numpy().reshape(num_samples * sequence_length, num_features)
    
    # If the data is normalized, inverse transform it to its original scale
    real_data_cpu = scaler.inverse_transform(real_data_cpu)
    generated_data_cpu = scaler.inverse_transform(generated_data_cpu)
    
    # Flatten for t-SNE
    real_data_cpu = real_data_cpu.reshape(num_samples, -1)
    generated_data_cpu = generated_data_cpu.reshape(num_samples, -1)
    
    # Combine the real and generated data for joint t-SNE visualization
    combined_data = np.concatenate([real_data_cpu, generated_data_cpu], axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(combined_data)
    
    # Split the results back into real and generated data
    real_tsne = tsne_results[:num_samples]
    generated_tsne = tsne_results[num_samples:]
    
    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], label='Real Data', alpha=0.6, color='blue')
    plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], label='Generated Data', alpha=0.6, color='red')
    plt.legend(loc='best')
    plt.title('t-SNE Visualization of Real vs Generated Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
