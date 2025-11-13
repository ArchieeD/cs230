import torch
from PIL import Image
from transformers import (
    Trainer,
    TrainingArguments,
)
from dataloader import  prepare_dataset, get_model_and_data
import os
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.nn import functional as F

class VisionLanguageDataCollator:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        self.pad_id = (
            processor.tokenizer.pad_token_id
            if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None
            else 0
        )

    def __call__(self, features):
        device = next(self.model.parameters()).device
        if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
            raise RuntimeError(f"ERROR: GPU is required. Current model device: {device}")

        input_ids_list = []
        attn_masks_list = []
        pixel_values_list = []
        grid_list = []
        prompt_lens = []

        for f in features:
            image = f["image"]  
            user_content = f["conversations"][0]['value']
            user_content = user_content.replace('<image>', '').strip()
            assistant_content = f["gt"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content+f" Put the expression in between $ $ signs. Characters should be separated by 1 space. For every subscript and superscript, always wrap the index or exponent in curly braces, even if it is a single character. Return only the LaTeX expression."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"$$ {assistant_content} $$"}
                    ]
                }
            ]

            prompt_only = self.processor.apply_chat_template(
                messages[:-1],
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding=False,
            )
            full = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding=False,
            )

            prompt_lens.append(prompt_only["input_ids"].shape[1])
            input_ids_list.append(full["input_ids"].squeeze(0))
            attn_masks_list.append(full["attention_mask"].squeeze(0))
            pixel_values_list.append(full["pixel_values"])
            grid_list.append(full["image_grid_thw"])
            
            # if isinstance(pv, (list, tuple)):
            #     # If multiple visuals were produced, just take the first for this task
            #     t = _ensure_4d(pv[0])
            #     pixel_values_list.append(t)
            # elif isinstance(pv, torch.Tensor):
            #     pixel_values_list.append(_ensure_4d(pv))
            # else:
            #     raise ValueError(f"pixel_values has unexpected type: {type(pv)}")


        max_len = max(x.shape[0] for x in input_ids_list)
        def pad_1d(x, pad_val=0):
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len -x.shape[0],), pad_val, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x, self.pad_id) for x in input_ids_list])
        attention_mask = torch.stack([pad_1d(x, 0) for x in attn_masks_list])


        labels =input_ids.clone()
        labels[attention_mask== 0] = -100
        for i, p_len in enumerate(prompt_lens):
            labels[i, :p_len] =-100

        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        grid_thw = grid_thw = torch.stack([g.squeeze().view(-1).to(torch.long) for g in grid_list], dim=0)

        # sum_T = int(pixel_values.shape[0])
        # sum_expected = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        # print(f"[DEBUG] visual tokens: sum_T={sum_T}, expected={sum_expected}, D={pixel_values.shape[1]}")
        # if sum_T != sum_expected:
        #     raise RuntimeError(
        #         f"Mismatch between visual tokens and grid: sum_T={sum_T} vs sum_expected={sum_expected}; "
        #         f"first rows grid_thw={grid_thw[:4].tolist()}"
        #     )
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "labels": labels,
        }
        return batch

def apply_lora_to_qwen_text(model, r=16, alpha=32, dropout=0.05, bias="none", target_mlp=True):
    """
    Wraps the Qwen text decoder with LoRA adapters on attention (and optionally MLP).
    Assumes the vision tower is frozen elsewhere.
    """
    # Good default target modules for Qwen2.5-VL text side
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Helpful for training stability
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_model = get_peft_model(model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model


def fine_tune_model(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir="qwen-finetuned",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    warmup_ratio=0.06,
    save_steps=200,
    eval_steps=200,
    logging_steps=100,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    max_grad_norm=1.0,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
):
   
    print("="*60)
    print("Starting Fine-tuning")
    print("="*60)
    
    model, _, processor, _, train_ds, val_ds, _ = get_model_and_data(model_id)
    
    for p in model.model.visual.parameters():
        p.requires_grad = False

    model = apply_lora_to_qwen_text(
        model,
        r=16,       # try 8 or 16
        alpha=32,   # typically 2x r
        dropout=0.05,
        bias="none",
        target_mlp=True,  # set False to only LoRA attention
    )


    # print("\nPreparing training dataset...")
    # train_dataset = prepare_dataset(train_ds, test=False)
    
    # print("\nPreparing validation dataset...")
    # val_dataset = prepare_dataset(val_ds, test=False)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, os.path.basename(output_dir))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=remove_unused_columns,
        report_to=report_to,
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    data_collator = VisionLanguageDataCollator(processor, model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    
    print("\n" + "="*60)
    print("Training started...")
    print("="*60)
    trainer.train()
    
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trainer.save_model()
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir) 
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)
    
    return trainer, output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on CROHME data.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen-finetuned",
        help="Directory (relative or absolute) to store the fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train and eval batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fine_tune_model(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

