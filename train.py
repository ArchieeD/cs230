import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 1) load your CROHME-rendered image (local, not URL)
img = Image.open("/path/to/crohme/img_0001.png").convert("RGB")

# 2) get vision inputs ONLY
vision_inputs = processor(images=img, return_tensors="pt").to(model.device)

# 3) run the vision tower to get features
with torch.no_grad():
    # different Qwen builds name this slightly differently;
    # many expose the vision encoder via model.vision_model or model.get_vision_tower()
    vision_outputs = model.vision_model(**vision_inputs)
    # this is usually [batch, seq_len, hidden_size]
    enc_feats = vision_outputs.last_hidden_state.squeeze(0)   # [L, C]
