from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

class QwenVisionEncoder:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

    @torch.no_grad()
    def __call__(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)
        vis_out = self.model.vision_model(**inputs)
        feats = vis_out.last_hidden_state.squeeze(0)
        return feats
