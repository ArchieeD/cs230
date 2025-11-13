import torch
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
)
from dataloader import load_data, split_dataset, save_img
import os
from tqdm import tqdm
import re

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
        # Check first parameter's device
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
    
    if model_device.type != 'cuda':
        raise RuntimeError(f"ERROR: Model is not on GPU! Model device: {model_device}")
    
    print(f"Loading dataset: {subset}")
    ds = load_data(subset)
    train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print(f"Train set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Test set: {len(test_ds)} samples")

    return model, model_device, processor, ds, train_ds, val_ds, test_ds

def run_baseline(model_id="Qwen/Qwen2.5-VL-3B-Instruct", output_file="baseline_results.txt"):
    """
    Run Qwen2.5-VL-3B-Instruct on the test set without any modifications.
    """
    # Set output_file to be in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, os.path.basename(output_file))
    
    model, model_device, processor, _, _, _, test_ds = get_model_and_data()

    results = []
    accuracy = 0
    
    for i in tqdm(range(len(test_ds))):
        sample = test_ds[i]
        image = sample["image"]
        conversation = sample["conversations"][0]['value']
        gt = sample["gt"]

        text_part = conversation.replace('<image>', '').strip()
        
        # Format messages with image in content list
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_part+f" Put the expression in between $ $ signs. Characters should be separated by 1 space. For every subscript and superscript, always wrap the index or exponent in curly braces, even if it is a single character. Return only the LaTeX expression."}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(model_device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_ids_only = generated_ids[0, input_length:]
        generated_text = processor.decode(
            generated_ids_only, 
            skip_special_tokens=True
        )

        match = re.search(r"\$\$(.*?)\$\$", generated_text)
        if match:
            pure_expression = match.group(1).replace(" ", "")
        else:
            # Fallback: try single $ signs or use the whole text
            match = re.search(r"\$(.*?)\$", generated_text)
            if match:
                pure_expression = match.group(1).replace(" ", "")
            else:
                pure_expression = generated_text.replace(" ", "")
        
        gt_original = gt
        gt = gt.replace(" ", "")
        
        results.append({
            "id": sample.get("id", i),
            "ground_truth": gt,
            "prediction": pure_expression
        })

        accuracy += gt == pure_expression
        
        
    

    with open(output_file, "w") as f:
        f.write("ID\tGround Truth\tPrediction\n")
        for r in results:
            f.write(f"{r['id']}\t{r['ground_truth']}\t{r['prediction']}\n")
        if len(results) > 0:
            f.write(f'Accuracy: {accuracy/len(results)}\n')
        else:
            f.write('Accuracy: N/A (no results)\n')
    
    print(f"\nResults saved to {output_file}")
    if len(results) > 0:
        print(f"\nAccuracy: {accuracy/len(results)}")
    else:
        print("\nAccuracy: N/A (no results)")
    # print(f"Processed {len(results)} test samples")
    
    return results, accuracy

def test_single_example(test_idx, model_id="Qwen/Qwen2.5-VL-3B-Instruct", subset="crohme2023"):
    """
    Test a single example from the test set to verify the pipeline works.
    """
    model, model_device, processor, _, _, _, test_ds = get_model_and_data()
    
    # Get first example
    print("\n" + "="*60)
    print("Testing first example from test set:")
    print("="*60)
    
    sample = test_ds[test_idx]
    print(f"\nSample ID: {sample.get('id', 'N/A')}")
    print(f"Conversations: {sample['conversations']}")
    print(f"Ground truth: {sample['gt']}")
    print(f"Image type: {type(sample['image'])}")
    print(f"Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
    
    image = sample["image"]
    conversation = sample["conversations"][0]['value']
    gt = sample["gt"]

    save_img(image)
    
    # Extract text part (remove <image> token if present)
    # Qwen2.5-VL expects content as a list with image and text
    text_part = conversation.replace('<image>', '').strip()

    
    # Format messages with image in content list
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_part+f" Put the expression in between $ $ signs. Characters should be separated by 1 space. For every subscript and superscript, always wrap the index or exponent in curly braces, even if it is a single character. Return only the LaTeX expression."}
            ]
        }
    ]

    
    
    # Prepare inputs
    print("\nProcessing inputs...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)
  
    
    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # Decode - extract only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_ids_only = generated_ids[0, input_length:]
    generated_text = processor.decode(
        generated_ids_only, 
        skip_special_tokens=True
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Ground Truth: {gt}")
    print(f"Prediction:   {generated_text}")
    print("="*60)
    
    return {
        "id": sample.get("id", 0),
        "ground_truth": gt,
        "prediction": generated_text
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_example(int(sys.argv[2]))
    else:
        run_baseline()

