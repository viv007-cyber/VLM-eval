# src/vlm_eval/diagnostics/generate_counterfactuals.py
import os
import glob
import pandas as pd
from google import genai
from google.genai import types

# Initialize the Gemini Client 
# Make sure your environment variable is set: export GEMINI_API_KEY="your-key"
client = genai.Client()

SYSTEM_PROMPT = """Given the image context implicitly captured in the question and answer, your task is to:
1. Generate an accurate <Description 1> which can be used for answering the question correctly without using the image.
2. Generate a wrong description <Description 2> which can be used for answering the question with a completely wrong answer <answer 2> without using the image.
3. Make sure both descriptions are sound and concise.
4. The wrong description’s sentence structure should be similar to the correct description.

Please output the statements strictly in this exact format:
Description 1: <Description 1>
Description 2: <Description 2>
Answer 2: <answer 2>"""

def consolidate_image_records(csv_dir: str) -> dict:
    """
    Scans all CSV files in a directory and groups question/answer records 
    by their primary visual image key.
    """
    image_map = {}
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Ensure unified expected headers exist
            required_cols = {'image_id', 'question', 'medgemma_answer'}
            if not required_cols.issubset(df.columns):
                continue
                
            for _, row in df.iterrows():
                img_id = str(row['image_id']).strip()
                if img_id not in image_map:
                    image_map[img_id] = []
                
                image_map[img_id].append({
                    "question": row['question'],
                    "medgemma_answer": row['medgemma_answer'],
                    "source_domain": os.path.basename(file_path).replace(".csv", "")
                })
        except Exception as e:
            print(f"Skipping unreadable/corrupt file {file_path}: {e}")
            
    return image_map

def query_gemini_counterfactual(question: str, answer: str) -> str:
    """
    Dispatches the structured QA combination to Gemini using the 
    specified system instructions.
    """
    user_content = f"Question: {question}\nAnswer: {answer}"
    
    try:
        # Utilizing gemini-3.5-flash for cost-efficient, lightning-fast text mutations
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,  # Keeping temperature low forces tighter structure mapping
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"API Network/Call Error: {e}")
        return ""