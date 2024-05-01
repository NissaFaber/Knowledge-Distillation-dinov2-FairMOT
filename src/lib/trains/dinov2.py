import torch
from transformers import AutoFeatureExtractor, AutoModel
# from transformers import Dinov2Config, Dinov2Model

from transformers import Dinov2Config, Dinov2Model, AutoImageProcessor
import torch
from torch import nn


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)



class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        # Define model configuration
        config = Dinov2Config(
            reshape_hidden_states=True,  # Ensure feature maps are reshaped
            output_hidden_states=True,   # Ensure all hidden states are output
            # return_dict=True            # Ensure outputs are returned as a dict
        )
        self.opt = opt
        model_name = "facebook/dino-vitb8"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Dinov2Model(config)
        self.model.eval()  # Set the model to evaluation mode
        self.device = torch.device(opt.device)
        self.model.to(self.device)

    def forward(self, batch_images):
        # Prepare the images using the feature extractor
        inputs = self.feature_extractor(images=batch_images['input'], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)

        # Access the reshaped hidden states; they should be in the desired 4D format
        features = outputs.last_hidden_state
        
        print(outputs, 'outputs bruv')
        return features


import torch
import torch.nn as nn
import torch.nn.functional as F

class DINO2HRNetAdapter(nn.Module):
    def __init__(self, hidden_size=768, target_shape=(270, 152, 272)):
        super().__init__()
        self.target_channels, self.target_height, self.target_width = target_shape
        
        # Assuming the hidden_size is compatible with a square-like reshaping
        self.intermediate_size = int((hidden_size ** 0.5) ** 2)
        self.reshape_conv = nn.Conv2d(hidden_size, self.target_channels, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_height, self.target_width))

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        # Drop class token if present (assuming it's the first token)
        x = x[:, 1:, :]
        
        # Reshape: [batch, hidden_size, sqrt(seq_len), sqrt(seq_len)]
        batch_size, seq_len, hidden_size = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, hidden_size, int(seq_len**0.5), int(seq_len**0.5))
        
        # Convert channel dimensions and resize
        x = self.reshape_conv(x)
        x = self.adaptive_pool(x)

        return x

# Example initialization and forward pass
adapter = DINO2HRNetAdapter()
dino_outputs = torch.randn(10, 197, 768)  # Example batch of outputs from DINOv2
hrnet_compatible_outputs = adapter(dino_outputs)
print(hrnet_compatible_outputs.shape)  # Should print torch.Size([10, 270, 152, 272])

