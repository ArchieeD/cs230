from datasets import load_dataset
import sys
import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def load_data(subset):
    # ds = load_dataset(
    #     "phxember/Uni-MuMER-Data",
    #     subset,
    #     cache_dir="~/cs230/crohme"
    # )
    # return ds["train"]
    ds = load_dataset("andito/mathwriting-google", cache_dir = "~/cs230/mathwriting")

    return ds, ds['train'], ds['validation'], ds['test']

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

def prepare_dataset(dataset, test=False, single_example=False, sample_idx=None):
    """
    Prepare dataset for training by converting samples to the message format.
    """
    def process_sample(sample, test=False):
        image = sample["image"]
        conversation = sample["conversations"][0]['value']
        gt = sample["gt"]
        
        text_part = conversation.replace('<image>', '').strip()
        
        if test:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_part + " Put the expression in between $ $ signs. Characters should be separated by 1 space. For every subscript and superscript, always wrap the index or exponent in curly braces, even if it is a single character. Return only the LaTeX expression."}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_part + " Put the expression in between $ $ signs. Characters should be separated by 1 space. For every subscript and superscript, always wrap the index or exponent in curly braces, even if it is a single character. Return only the LaTeX expression."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": f"$$ {gt} $$"
                }
            ]
        
        return {
            "messages": messages,
            "image": image,
            "gt": gt
        }
    if single_example:
        return process_sample(dataset[sample_idx], test=test)
    else:
        return dataset.map(lambda x: process_sample(x, test=test), remove_columns=dataset.column_names)

def get_model_and_data(model_id="Qwen/Qwen2.5-VL-3B-Instruct", gpu=True, subset="crohme2023"):
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("ERROR: No GPU available. This script requires a GPU to run.")

    device = torch.device("cuda")
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Verify model is on GPU - error out if not
    if hasattr(model, 'device'):
        model_device = model.device
        print(f"Model device: {model_device}")
    else:
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
    
    if model_device.type != 'cuda':
        raise RuntimeError(f"ERROR: Model is not on GPU! Model device: {model_device}")
    
    print(f"Loading dataset: {subset}")
    # ds = load_data(subset)
    # train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    ds, train_ds, val_ds, test_ds = load_data(subset)
    print(f"Train set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Test set: {len(test_ds)} samples")

    return model, model_device, processor, ds, train_ds, val_ds, test_ds

#FIXME
def save_img(img):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = os.path.join(current_dir, "crohme")
    save_dir = os.path.join(current_dir, "mathwriting")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sample_image.png")
    img.save(save_path)


if __name__ == "__main__":
    ds = load_data("crohme2023")
    features, rows = get_ds_info(ds)
    sample = ds[np.random.randint(0, rows)]
    img = sample["image"]
    latex = sample["gt"]
    print(f"Image saved")
    print(f"LaTeX: {latex}")
