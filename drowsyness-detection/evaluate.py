import os
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
from datasets import load_dataset
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np

# Load our local model or Hugging Face cached version
try:
    model_path = hf_hub_download(repo_id='mosesb/drowsiness-detection-yolo-cls', filename='best.pt')
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Let's load a subset of the dataset
print("Loading dataset from Hugging Face...")
# We use streaming=False, we just load 150 items from the train split to evaluate quickly
dataset = load_dataset("akahana/Driver-Drowsiness-Dataset", split="train[:150]")

print("Format of dataset features:", dataset.features)
label_names = dataset.features["label"].names if hasattr(dataset.features.get("label"), "names") else None
print("Label names in the dataset:", label_names)

correct = 0
total = 0
drowsy_correct = 0
drowsy_total = 0
nondrowsy_correct = 0
nondrowsy_total = 0

print("Starting evaluation...")
for item in dataset:
    image = item["image"]
    true_label_idx = item["label"]
    
    # Try to map true_label to 'Drowsy' or 'Non Drowsy' based on the dataset's label names
    if label_names:
        true_label_str = label_names[true_label_idx]
        # the dataset classes might be 'Alert' and 'Drowsy', or similar.
    else:
        # Fallback just in case
        true_label_str = str(true_label_idx)

    # Let's see what YOLO predicts
    results = model.predict(image, verbose=False)
    probs = results[0].probs
    top1_class_index = probs.top1
    pred_class_name = model.names[top1_class_index]
    
    # Align labels
    # Model pred_class_name is likely 'Drowsy' or 'Non Drowsy'
    # 'true_label_str' might be 'Drowsy' or 'Non Drowsy' or 'Alert'
    
    is_drowsy_pred = 'drowsy' in pred_class_name.lower() and 'non' not in pred_class_name.lower()
    is_drowsy_true = 'drowsy' in true_label_str.lower() and 'non' not in true_label_str.lower()
    
    # Just in case it's numeric and 1=Drowsy, we'll wait and see. 
    # Let's just match string logic:
    if is_drowsy_true:
        drowsy_total += 1
        if is_drowsy_pred:
            drowsy_correct += 1
            correct += 1
    else:
        nondrowsy_total += 1
        if not is_drowsy_pred:
            nondrowsy_correct += 1
            correct += 1
            
    total += 1
    
    if total % 50 == 0:
        print(f"Processed {total} images...")

print("\n--- Evaluation Results (150 Images) ---")
print(f"Total Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
if drowsy_total > 0:
    print(f"Drowsy Accuracy (Recall for Drowsy): {drowsy_correct/drowsy_total*100:.2f}% ({drowsy_correct}/{drowsy_total})")
if nondrowsy_total > 0:
    print(f"Non-Drowsy Accuracy (Recall for Non-Drowsy): {nondrowsy_correct/nondrowsy_total*100:.2f}% ({nondrowsy_correct}/{nondrowsy_total})")

