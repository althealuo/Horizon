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


from scipy.stats import ttest_rel, sem
# ---------------------------------------------------
# 1. Helper: compute mean and 95% CI
# ---------------------------------------------------
def mean_ci(data, confidence=0.95):
    """Return (mean, lower_CI, upper_CI) for a list or array."""
    data = np.array(data)
    m = np.mean(data)
    s = sem(data)
    h = 1.96 * s
    return m, m - h, m + h


# ---------------------------------------------------
# 2. Extract final accuracy values for all models
# ---------------------------------------------------
def extract_final_metrics(combined_outputs):
    """
    combined_outputs: dict with model_name -> {metrics}
    Returns a DataFrame with final_epoch accuracy for each model.
    """
    rows = []

    for model_name, data in combined_outputs.items():
        final_idx = data["final_epoch"] - 1

        rows.append({
            "model": model_name,
            "acc_overall": data["test_acc_prog"][final_idx],
            "acc_h1": data["test_acc_h1_prog"][final_idx],
            "acc_h6": data["test_acc_h6_prog"][final_idx]
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------
# 3. Compute stats (mean CI + paired t-test)
# ---------------------------------------------------
def compute_stats(df):
    """
    df: DataFrame from extract_final_metrics()

    Returns:
        stats_dict: dictionary of results
        and prints clean summary text
    """
    h1_vals = df["acc_h1"].values
    h6_vals = df["acc_h6"].values

    # Mean ± CI
    h1_mean, h1_low, h1_high = mean_ci(h1_vals)
    h6_mean, h6_low, h6_high = mean_ci(h6_vals)

    # Two-sided paired t-test
    t_stat, p_two = ttest_rel(h1_vals, h6_vals)

    # One-sided directional hypothesis: H1 > H6
    if t_stat > 0:
        p_one = p_two / 2
    else:
        p_one = 1 - p_two / 2

    stats_dict = {
        "h1_mean": h1_mean,
        "h1_CI": (h1_low, h1_high),
        "h6_mean": h6_mean,
        "h6_CI": (h6_low, h6_high),
        "t_stat": t_stat,
        "p_two_sided": p_two,
        "p_one_sided": p_one,
        "model_count": len(df)
    }

    # Pretty print summary
    print("\n===== Horizon Condition Accuracy Stats =====")
    print(f"H1 mean accuracy = {h1_mean:.3f}  (95% CI [{h1_low:.3f}, {h1_high:.3f}])")
    print(f"H6 mean accuracy = {h6_mean:.3f}  (95% CI [{h6_low:.3f}, {h6_high:.3f}])")
    print("\nPaired t-test (H1 vs H6):")
    print(f"  t = {t_stat:.3f},  two-sided p = {p_two:.5f}")
    print(f"  One-sided p (H1 > H6) = {p_one:.5f}")
    print("===========================================\n")

    return stats_dict