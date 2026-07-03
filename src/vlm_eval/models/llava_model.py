# src/vlm_eval/models/llava_model.py
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlm_eval.models.base import BaseVLM
from vlm_eval.models import register_model

@register_model("llava")
class LlavaVLAdapter(BaseVLM):
    def load_model(self):
        print(f"Loading LLaVA from {self.model_path}...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            images=image, 
            text=text, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=32, 
                **kwargs
            )
            prompt_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            
            # Decode tensors back to strings
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
        return output_text[0].strip()