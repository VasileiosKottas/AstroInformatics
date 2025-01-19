import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# def find_cond_mask(np_data, targ_first, targ_last):
#     """
#     Description:
    
#     Creates a 'block-like' conditional mask; i.e a tensor with the same shape as the data (B x K x L) indicating the 
#     position of the conditional data points (1) and the non-conditional points (0) such as imputation targets
    
    
#     Parameters:
    
#     np_data (np.darray): The observed data
#     targ_first (int): The index at which the conditional block starts in the time dimension (i.e along L)
#     targ_last (int): The index at which the conditional block end in the time dimension (i.e along L)
    
    
#     """
    
#     cond_mask = np.ones_like(np_data)
#     cond_mask[:,:,targ_first:targ_last] = np.zeros((cond_mask.shape[0], cond_mask.shape[1], targ_last - targ_first))
#     return cond_mask

def df_concat(df_cond, df_imp):
    
    """
    Description:
    
    Combines two data frames and reorders their columns in ascending order. A mask is also created indicating which
    elment comes from the conditional data frame (1) and the imputation data frame (0)
    
    """
    
    df_combined = pd.concat([df_imp, df_cond], axis=1)
    df_combined = df_combined.reindex(sorted(df_combined.columns, key=lambda x: float(x)), axis = 1)

    cond_row = np.zeros(df_combined.shape[1], dtype=int)
    cond_columns = df_cond.columns
    cond_row[df_combined.columns.isin(cond_columns)] = 1
    
    cond_mask = np.tile(cond_row, (df_combined.shape[0], 1))
    
    return df_combined, cond_mask

class spectra_Dataset(Dataset):
  
    """
    Description:
    
    Converts an np_array into a Dataset object and becomes normalised in the time dimension. The __getitem__ allows the extraction of
    five torch tensors: observed_data, observed_mask, conditional_mask, groundtruth_mask and time points. There are no missing data 
    points in the dataset so the observed mask (indicating the positions of the observed values) is a tensor of ones. The data points
    from which the model makes predictions are given by the ones in conditional_mask. Ground_truth = Cond_mask. 
    
    
    Parameters:
    
    np_data (numpy.darray): The observed data
    cond_mask (numpy.darray): Conditional mask
    eval_length (int): The length of the time series
    target_dim (int): The number of features at each timepoint; target_dim = 1 for spectra as the only feature is flux.
    mode (string): Determines whether the train, test or validation set is the output.
    
    """
    
    
    def __init__(self,
                 np_data, 
                 cond_mask,
                 eval_length,   
                 target_dim,    
                 mode="train"):
        
        self.eval_length = eval_length
        self.target_dim = target_dim
        
        # train + val, test split:
        train_val_data, test_data = train_test_split(np_data, test_size=0.1, random_state=42)
        # train, val split:
        train_data, val_data = train_test_split(train_val_data, test_size=0.10, random_state=42)
        
        
        if mode == "train":
            self.observed_data = train_data
        elif mode == "valid":
            self.observed_data = val_data
        elif mode == "test":
            self.observed_data = test_data
        
        
        # Normalize each row (along each time series):
        
        self.mean = self.observed_data.mean(axis=1, keepdims=True)
        self.std = self.observed_data.std(axis=1, keepdims=True)
        self.observed_data = (self.observed_data - self.mean) / self.std
        
        
        #
        
        self.observed_data = self.observed_data.reshape(self.observed_data.shape[0], target_dim, eval_length)
        self.observed_mask = np.ones_like(self.observed_data)  # all 1s, shape (N x K x L)
        self.cond_mask = cond_mask.reshape(cond_mask.shape[0], target_dim, eval_length) 
        self.gt_mask = self.cond_mask 
        
    def __getitem__(self, index):
        s = {
            "observed_data": torch.tensor(self.observed_data[index], dtype=torch.float32),
            "observed_mask": torch.tensor(self.observed_mask[index], dtype=torch.float32),
            "cond_mask": torch.tensor(self.cond_mask[index], dtype=torch.float32),
            "gt_mask": torch.tensor(self.gt_mask[index], dtype=torch.float32),
            "timepoints": torch.tensor(np.arange(self.eval_length), dtype=torch.float32)
            }
        
        return s
    
    def __len__(self):
        return len(self.observed_data)
        
        
def get_dataloader(df_tot, cond_mask, batch_size, device, eval_length=128, target_dim=1):
    train_dataset = spectra_Dataset(df_tot, cond_mask, eval_length, target_dim, mode="train")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    
    
    test_dataset = spectra_Dataset(df_tot, cond_mask, eval_length, target_dim, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    valid_dataset = spectra_Dataset(df_tot, cond_mask, eval_length, target_dim, mode="valid")
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    
    scaler = torch.from_numpy(train_dataset.std).to(device).float() #std
    mean_scaler = torch.from_numpy(train_dataset.mean).to(device).float() #mean


    return train_loader, valid_loader, test_loader, scaler, mean_scaler