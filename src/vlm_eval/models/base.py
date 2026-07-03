# src/vlm_eval/models/base.py
#This file forces every model adapter you create later (like LLaVA or MedGemma) to use identical code rules.
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Union

class BaseVLM(ABC):
    """Abstract base class that all model interfaces must implement."""
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        self.model_path = model_path
        self.device = device
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Logic to initialize the model, tokenizer, and vision processors."""
        pass

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """Takes a single PIL Image and a text prompt, returns model response string."""
        pass