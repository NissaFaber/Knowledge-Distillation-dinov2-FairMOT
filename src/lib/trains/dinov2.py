import torch
from transformers import AutoFeatureExtractor, AutoModelnov2ForImageClassification

from PIL import Image
import requests
import torch


class Dinov2(Object):
  def __init__(self, opt):
    # Load the model and feature extractor
    model_name = "facebook/dino-vitb8"  # Use the correct model identifier
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    self.model = model
    self.model.eval()  # Set the model to evaluation mode
    device = torch.device(opt.device)
    self.model.to(device)
    
    # Function to process and extract features
    def extract_features(batch):
    
        # with torch.no_grad():
        #     for images in data_loader:
        #         # Assume images are directly in the correct format
        #         # If not, you need to preprocess them:
        #         # inputs = feature_extractor(images=images, return_tensors="pt")
        #         inputs = images.to(device)
    
        outputs = self.model(batch['input'])
        # Assuming the model outputs the desired features at a specific layer index
        # You might need to adjust this depending on the actual model architecture
        features = outputs.hidden_states[-1]  # Modify as needed to match the expected output layer
        
        # Optionally, resize features to the expected output shape if needed
        features = torch.nn.functional.interpolate(features, size=(152, 272), mode='bilinear', align_corners=False)
        
        all_features.append(features)
    
        return torch.cat(all_features, dim=0)

# Extract features
# features = extract_features(train_loader, model, device)
# print(features.shape)  # Debugging line to check output shape

    
