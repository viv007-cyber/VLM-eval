# src/vlm_eval/models/paligemma_model.py
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from vlm_eval.models.base import BaseVLM
from vlm_eval.models import register_model

@register_model("paligemma")
class PaliGemmaAdapter(BaseVLM):
    def load_model(self):
        print(f"Loading PaliGemma from {self.model_path}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        # PaliGemma directly takes the image and text arguments in its processor
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                **kwargs
            )
            
            # Trim the prompt tokens out of the sequence array
            input_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
        return output_text[0].strip()