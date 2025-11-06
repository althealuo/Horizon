import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_df():
    df = pd.read_csv("./data/my_horizon_data_all.csv", dtype={"subject": str})
    return df

def get_dataloaders(X_seq, X_static, y, SEQ_LEN=2, BATCH_SIZE=32, TIME_STEPS=4):
    
    X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(X_seq, X_static, y, test_size=0.2, random_state=42)


    # split based on original data frame
    h1_mask = X_static_test['gameLength'] == 1
    h6_mask = X_static_test['gameLength'] == 6
    X_static_test_raw = X_static_test.copy()

    scaler = StandardScaler()
    X_seq_train = scaler.fit_transform(X_seq_train)
    X_seq_test = scaler.transform(X_seq_test)
    X_static_train = scaler.fit_transform(X_static_train)
    X_static_test = scaler.transform(X_static_test)

    # reshape to (num_samples, time_steps, features)
    X_seq_train = X_seq_train.reshape(-1, TIME_STEPS, SEQ_LEN)
    X_seq_test = X_seq_test.reshape(-1, TIME_STEPS, SEQ_LEN)

    
    X_seq_train_tensor = torch.tensor(X_seq_train, dtype=torch.float32) # sklearn output float64, doesn't work with torch
    X_seq_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32) 

    X_static_train_tensor = torch.tensor(X_static_train, dtype=torch.float32) 
    X_static_test_tensor = torch.tensor(X_static_test, dtype=torch.float32) 

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long) # pandas series to tensor
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_seq_train_tensor, X_static_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_seq_test_tensor, X_static_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    h1_mask_bool = torch.tensor(h1_mask.to_numpy(), dtype=torch.bool)
    h6_mask_bool = torch.tensor(h6_mask.to_numpy(), dtype=torch.bool)

    X_seq_test_h1 = torch.tensor(X_seq_test[h1_mask_bool], dtype=torch.float32)
    X_seq_test_h6 = torch.tensor(X_seq_test[h6_mask_bool], dtype=torch.float32)

    X_static_test_h1 = torch.tensor(X_static_test[h1_mask_bool], dtype=torch.float32)
    X_static_test_h6 = torch.tensor(X_static_test[h6_mask_bool], dtype=torch.float32)

    y_test_h1 = y_test_tensor[h1_mask_bool]
    y_test_h6 = y_test_tensor[h6_mask_bool]

    
    test_loader_h1 = DataLoader(TensorDataset(X_seq_test_h1, X_static_test_h1, y_test_h1), batch_size=BATCH_SIZE, shuffle=False)
    test_loader_h6 = DataLoader(TensorDataset(X_seq_test_h6, X_static_test_h6, y_test_h6), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, test_loader_h1, test_loader_h6


def save_output(model_dict, filename="model_results"):
    dump_history = {}

    for model_name, content in model_dict.items():
        # content = {"model": <model_object>, "train_acc": ..., "test_acc": ..., ...}
        filtered = {k: v for k, v in content.items() if k != "model"}
        dump_history[model_name] = filtered
    # store the outputs 
    import json
    with open(f"{filename}.json", "w") as f:
        json.dump(dump_history, f)
        print(f"Saved model results to {filename}.json")
        