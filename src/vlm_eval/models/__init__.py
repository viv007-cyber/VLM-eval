# src/vlm_eval/models/__init__.py
from typing import Dict, Type
from vlm_eval.models.base import BaseVLM

# A registry dictionary to map string names to model classes dynamically
MODEL_REGISTRY: Dict[str, Type[BaseVLM]] = {}

def register_model(name: str):
    """Decorator to easily register a new VLM."""
    def decorator(cls: Type[BaseVLM]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# Lazy-loaded imports to prevent dependency crashes
def get_model(name: str, model_path: str, **kwargs) -> BaseVLM:
    if name not in MODEL_REGISTRY:
        # Import your model files here so they self-register when called
        if name == "qwen":
            import vlm_eval.models.qwen_vl
        elif name == "llava":
            import vlm_eval.models.llava_model
        elif name == "instructblip":
            import vlm_eval.models.instructblip
        elif name == "medgemma":
            import vlm_eval.models.medgemma
        elif name=="paligemma":
            import vlm_eval.models.paligemma
            
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered. Available: {list(MODEL_REGISTRY.keys())}")
        
    return MODEL_REGISTRY[name](model_path, **kwargs)

