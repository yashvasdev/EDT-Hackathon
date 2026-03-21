from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Load the model from the Hugging Face Hub
model_path = hf_hub_download(repo_id='mosesb/drowsiness-detection-yolo-cls', filename='best.pt')
model = YOLO(model_path)

# Run inference on an image
image_path = r"drowsyness-detection\testData\image.png"
results = model.predict(image_path)

# Print the top prediction
probs = results[0].probs
top1_class_index = probs.top1
top1_confidence = probs.top1conf
class_name = model.names[top1_class_index]

print(f"Prediction: {class_name} with confidence {top1_confidence:.4f}")
