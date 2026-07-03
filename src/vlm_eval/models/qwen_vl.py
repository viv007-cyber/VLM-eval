# src/vlm_eval/models/qwen_model.py
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from vlm_eval.models.base import BaseVLM
from vlm_eval.models import register_model

@register_model("qwen")
class QwenVLAdapter(BaseVLM):
    def load_model(self):
        print(f"Loading Qwen2.5-VL from {self.model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            min_pixels=256 * 256,       # Lower bounds patch limit
            max_pixels=1024 * 1024,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        if "temperature" in kwargs and kwargs["temperature"] == 0.0:
            kwargs.pop("temperature")
            kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32, **kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text[0]

