# src/vlm_eval/models/medgemma_model.py
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from vlm_eval.models.base import BaseVLM
from vlm_eval.models import register_model

@register_model("medgemma")
class MedGemmaAdapter(BaseVLM):
    def load_model(self):
        print(f"Loading MedGemma from {self.model_path}...")
        if "27b" in self.model_path.lower():
            print("27B Variant detected: Activating 4-bit quantization layout for VRAM constraints.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",               
                bnb_4bit_use_double_quant=True,         
                bnb_4bit_compute_dtype=torch.bfloat16    
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                quantization_config=quantization_config, # <-- Pass the config object here
                device_map="auto"
            )

        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
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
        
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                **kwargs
            )
            
            input_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
        return output_text[0].strip()


        