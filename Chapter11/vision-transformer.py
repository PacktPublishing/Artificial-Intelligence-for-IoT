from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Load a pretrained ViT model and feature extractor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load and preprocess an image
image = Image.open("sample_image.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
