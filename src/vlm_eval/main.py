# src/vlm_eval/main.py
import os
import argparse
import json
import gc
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from vlm_eval.models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Multi-Domain VLM Evaluator")
    parser.add_argument("--model_type", type=str, required=True, help="e.g., qwen, llava, medgemma")
    parser.add_argument("--model_path", type=str, required=True, help="HF path or local weights directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to your image subset folder")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to specific domain JSON config")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Open Domain Config Safely
    with open(args.prompt_file, "r") as f:
        config_data = json.load(f)
    
    domain_name = config_data["domain"]
    prompts_dict = config_data["prompts"]

    # Gather target images recursively across nested subdirectories
    valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    image_paths = []
    for p in Path(args.image_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            image_paths.append(p)
            
    print(f"Loaded {len(image_paths)} images from '{args.image_dir}' for evaluation.")
    
    model = get_model(args.model_type, args.model_path)

    records = []
    
    # Track which modalities are processed in this specific run for file naming down the line
    seen_modalities = set()
    
    print(f"Starting Evaluation Run | Model: {args.model_type} | Domain: {domain_name}")
    
    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Evaluating {domain_name}"), start=1):
        img_name = img_path.name
        path_parts = [part.lower() for part in img_path.parts]
        
        # 2. Identify Modality dynamically
        modality = "Unknown"
        if "mri" in path_parts:
            modality = "MRI"
        elif "xray" in path_parts or "x-ray" in path_parts:
            modality = "Xray"
            
        seen_modalities.add(modality.lower())
        
        # 3. Dynamically identify Anatomy Zone
        anatomy_zone = "Unknown"
        for zone in ["knee", "shoulder", "hip"]:
            if zone in path_parts:
                anatomy_zone = zone.capitalize()
                break
                
        # 4. Dynamically identify Ground Truth Pathology
        parent_folder = img_path.parent.name
        if parent_folder.lower() in [anatomy_zone.lower(), modality.lower(), "result", "datasets"]:
            ground_truth_label = "general_anatomy"
        else:
            ground_truth_label = parent_folder 
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping image {img_name}: {e}")
            continue

        for prompt_id, prompt_text in prompts_dict.items():
            # Evaluation run inference step
            response = model.generate(image, prompt_text, temperature=args.temperature)
            
            records.append({
                "image_name": img_name,
                "modality": modality,               
                "anatomy": anatomy_zone,             
                "ground_truth": ground_truth_label,   
                "domain": domain_name,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "model_response": response,
                "temperature": args.temperature
            })
            
        # Periodic memory cleanup mimicking your working setup configuration loop
        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not records:
        print("No evaluation records were generated.")
        return

    # 5. Build Dynamic Variant and Modality Specific Filenames
    model_name_tag = args.model_type
    if "medgemma" in args.model_type.lower():
        if "27b" in args.model_path.lower():
            model_name_tag = "medgemma_27b"
        else:
            model_name_tag = "medgemma_4b"
            
    # Determine the modality label based on what the loop actually processed
    if "mri" in seen_modalities and "xray" in seen_modalities:
        modality_tag = "mixed"
    elif "mri" in seen_modalities:
        modality_tag = "mri"
    elif "xray" in seen_modalities:
        modality_tag = "xray"
    else:
        modality_tag = "unknown_modality"

    # Save Domain-Specific Dataframe
    df = pd.DataFrame(records)
    os.makedirs("outputs", exist_ok=True)
    
    # Combined target string pattern: outputs/{model}_{modality}_{domain}.csv
    output_filename = f"{model_name_tag}_{modality_tag}_{domain_name}.csv"
    output_path = os.path.join("outputs", output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"Success! Separate domain evaluation file saved to: {output_path}")

if __name__ == "__main__":
    main()