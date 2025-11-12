from datasets import load_dataset
import sys
import os
import numpy as np

def load_data(subset):
    ds = load_dataset(
        "phxember/Uni-MuMER-Data",
        subset,
        cache_dir="~/cs230/crohme"
    )
    return ds["train"]

def split_dataset(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        ds: HuggingFace dataset
        train_ratio: proportion for training set
        val_ratio: proportion for validation set
        test_ratio: proportion for test set
        seed: random seed for reproducibility
    
    Returns:
        train_ds, val_ds, test_ds
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    ds = ds.shuffle(seed=seed)
    total_size = len(ds)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, train_size + val_size))
    test_ds = ds.select(range(train_size + val_size, total_size))
    
    return train_ds, val_ds, test_ds

def get_ds_info(ds):
    print(ds)
    features = ds.features
    rows = len(ds)
    return features, rows

if __name__ == "__main__":
    ds = load_data("crohme2023")
    features, rows = get_ds_info(ds)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "crohme")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sample_image.png")
    sample = ds[np.random.randint(0, rows)]
    img = sample["image"]
    latex = sample["gt"]
    img.save(save_path)
    print(f"Image saved to {save_path}")
    print(f"LaTeX: {latex}")
