import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from main_model_spectra import CSDI_spectra
from spectra_dataset import get_dataloader, df_concat, spectra_Dataset
from utils_spectra import train, evaluate 
      
def train_model(observed_data, cond_mask):
    """
    Desc:
    
    Loads hyper-parameters, data and trains model.
    
    Paramaters:
    
    observed_data (numpy.ndarray): np array of shape N x L; N = number of spectra (10000), L = length of time series 
    cond_mask (numpy.ndarray): np array of shape N x L; 1s correspond to conditional data, 0s to imputation targets
    
    """
    #loading hyper-parameters
    def load_config(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    config = load_config('base_spectra.yaml')
    batch_size = config["train"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_length = observed_data.shape[-1]
    target_dim = 1
    
    #creating dataloader and model
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        observed_data, cond_mask, batch_size, device,eval_length=eval_length, target_dim=target_dim)
    model = CSDI_spectra(config, device, target_dim=1)
    model.to(device)
    
    #training
    train(model, config, train_loader, valid_loader = valid_loader, valid_epoch_interval = 5, foldername= "output_samples")
    
    return model, valid_loader


def eval_model(model, valid_loader, num_samp = 50):
    """
    Desc:
    
    Sample imputation (reverse difussion process)
    
    Parameters:
    
    model: trained model
    valid_loader: data loader for validation data
    num_samp:  number of imputed samples which are then averaged out
    
    """
    model.eval()
    
    all_samples = []
    all_observed_data = []
    all_target_mask = []
    all_observed_mask = []
    
    with torch.no_grad():  
        for batch in valid_loader:
            print(batch)
            x = model.evaluate(batch, num_samp)
            samples = x[0]
            observed_data = x[1]
            target_mask = x[2]
            observed_mask = x[3]

            all_samples.append(samples)
            all_observed_data.append(observed_data)
            all_target_mask.append(target_mask)
            all_observed_mask.append(observed_mask)
    
    # Concatenate all batches along the batch dimension (dim=0)
    all_samples = torch.cat(all_samples, dim=0)
    all_observed_data = torch.cat(all_observed_data, dim=0)
    all_target_mask = torch.cat(all_target_mask, dim=0)
    all_observed_mask = torch.cat(all_observed_mask, dim=0)

    # Averaging over num_samp imputed samples
    all_samples = all_samples.mean(dim=1)

    # Converting tensors to numpy arrays
    all_samples = all_samples.to(torch.device("cpu")).numpy()
    all_observed_data = all_observed_data.to(torch.device("cpu")).numpy()
    all_target_mask = all_target_mask.to(torch.device("cpu")).numpy()
    all_observed_mask = all_observed_mask.to(torch.device("cpu")).numpy()

    return all_samples, all_observed_data, all_target_mask, all_observed_mask


def eval_test_plotter(eval_output, cond_mask, sorted_wavelengths, valid_dataset, index=0, i=35, j=4):
    samples, observed_data, target_mask, observed_mask = eval_output
    observed_data_x = observed_data[index,0,:]
    samples_x = samples[index,0,:]
    
    size = valid_dataset.observed_data.shape[0]
    mean_valid = valid_dataset.mean.reshape(size)
    std_valid = valid_dataset.std.reshape(size)
    
    samples_x[cond_mask[0] == 1] = observed_data_x[cond_mask[0] == 1]
    
    samples_x = samples_x * std_valid[index] + mean_valid[index]
    observed_data_x = observed_data_x * std_valid[index] + mean_valid[index]
    samples_region = samples_x[i:-j]
    observed_data_region = observed_data_x[i:-j]
    
    sorted_wavelengths_region = sorted_wavelengths[i:-j]
    
    # Filter for wavelengths < 35
    mask = sorted_wavelengths_region < 35
    sorted_wavelengths_region = sorted_wavelengths_region[mask]
    samples_region = samples_region[mask]
    observed_data_region = observed_data_region[mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_wavelengths_region, samples_region, c="blue", linewidth=1)
    plt.plot(sorted_wavelengths_region, observed_data_region, c="red", linewidth=1)
    plt.xlabel("Wavelengths")
    plt.ylabel("Flux")
    plt.title("Sample vs Observed Data for Wavelengths < 35")
    plt.grid(True)
    plt.show()

    #plt.scatter(filtered_wavelengths, filtered_cond_data , c = "green", s=6)
    

def error_chi_squared(eval_output, cond_mask, valid_dataset, i = 35, j = 4):

    size = valid_dataset.observed_data.shape[0]
    mean_valid = valid_dataset.mean.reshape(size)
    std_valid = valid_dataset.std.reshape(size)
    
    chi_squared = np.zeros(size)
    counter = 0
    
    for index in range(size):
        samples_x = eval_output[0][index,0,:]
        observed_data_x = eval_output[1][index,0,:]
        samples_x[cond_mask[0] == 1] = observed_data_x[cond_mask[0] == 1]
        
        samples_x = samples_x * std_valid[index] + mean_valid[index]
        observed_data_x = observed_data_x * std_valid[index] + mean_valid[index]
        
        samples_region = samples_x[i:-j]
        observed_data_region = observed_data_x[i:-j]

        res = observed_data_region - samples_region
        chi_squared[index] = (1 / (len(samples_region) - 2)) * np.sum((res ** 2) / 
                                                                      ((0.1 * samples_region ) ** 2 + (0.1 *observed_data_region ) ** 2))
        
        if chi_squared[index] <= 5:
            counter += 1
        
    chi_under_5 = 100* counter/size
    
        # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(chi_squared, bins=30, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Chi-Squared Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Chi-Squared Values')
    plt.grid(True)
    plt.show()
    
    
    return chi_squared, chi_under_5




