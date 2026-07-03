# src/vlm_eval/models/instructblip_model.py
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from vlm_eval.models.base import BaseVLM
from vlm_eval.models import register_model

@register_model("instructblip")
class InstructBlipAdapter(BaseVLM):
    def load_model(self):
        print(f"Loading InstructBlip from {self.model_path}...")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = InstructBlipProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,  
                **kwargs
            )
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
        return generated_text[0].strip()