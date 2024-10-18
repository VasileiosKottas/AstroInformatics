import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


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
    
    def __init__(self,
                 np_data, 
                 
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
 
        self.observed_data = self.observed_data
        
        
    def __getitem__(self, index):
        s = torch.tensor(self.observed_data[index], dtype=torch.float32)
        return s
    
    def __len__(self):
        return len(self.observed_data)
        
        
def get_dataloader(df_tot, batch_size, device, eval_length=128, target_dim=1):
    train_dataset = spectra_Dataset(df_tot, eval_length, target_dim, mode="train")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    
    scaler = torch.from_numpy(train_dataset.std).to(device).float() #std
    mean_scaler = torch.from_numpy(train_dataset.mean).to(device).float() #mean

    return train_loader, scaler, mean_scaler

def use_data(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind_of_data = kwargs.get('data')
    df_inter = pd.read_csv('../data/interpolated_spectra.csv')
    df_spectra = pd.read_csv('../data/spectra.csv')
    if kind_of_data == 'photometry':
        df = df_inter
    else:
        df, mask = df_concat(df_inter,df_spectra)
    sorted_wavelengths = df.columns.values.astype(float)
    df_plot = df.T
    df = df.T.to_numpy()
    # df_spectra = df_spectra.T.to_numpy()
    # df = torch.tensor(df)
    df.shape
    print(df.shape)
    train_loader, scaler, mean_scaler = get_dataloader(df, 32, device,eval_length=223)
    for batch in train_loader:
        print(batch)  # Should be (batch_size, seq_len, 1)
        break
    x = train_loader
   
    return df, sorted_wavelengths