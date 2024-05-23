

import torch
from torch import nn
from transformers import ViTImageProcessor, Dinov2Model, Dinov2Config, AutoBackbone
from torch.functional import F



dinov2_models = {'base': ['facebook/dinov2-base', 768] ,
                 'small': ['facebook/dinov2-small', 384 ],
                 'large' : ['facebook/dinov2-large', 1024],
                 'giant' : ['facebook/dinov2-giant', 1536]}




class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        self.device = torch.device(opt.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(dinov2_models[opt.dinov2][0], size={'height': 1088, 'width': 608}, do_rescale=False, reshape_hidden_states=True)
        config = Dinov2Config.from_pretrained(dinov2_models[opt.dinov2][0], reshape_hidden_states=True, output_hidden_states=True)
        self.model = AutoBackbone.from_config(config).to(self.device)
        # print(self.model.config)
        # self.model = Dinov2Model.from_pretrained('facebook/dinov2-small')
        
        self.model.eval()

    def forward(self, batch_images):
        inputs = self.feature_extractor(images=batch_images['input'], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print(inputs['pixel_values'].shape)

        outputs = self.model(**inputs)
        features = outputs[0][0]

        # Access positional embeddings
        # pos_embeddings = self.model.embeddings.position_embeddings

        # print(pos_embeddings)
        # print(features.shape)

        return features



class DINO2HRNetAdapter(nn.Module):
    def __init__(self, opt, target_shape=(270, 152, 272), device='0'):
        super().__init__()
        hidden_size = dinov2_models[opt.dinov2][1]
        self.target_channels, self.target_height, self.target_width = target_shape
        
        # Upsampling layer to increase resolution
        self.upsample = nn.Upsample(size=(152, 224), mode='bilinear', align_corners=False).to(device)  # Upsample to an intermediate size

        # Convolution to reduce channel depth and adjust to the target channel depth
        self.channel_adjust_conv = nn.Conv2d(hidden_size, self.target_channels, kernel_size=(1, 1), stride=1).to(device)

        # Adaptive pooling to match the exact target dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_height, self.target_width)).to(device)

    def forward(self, x):
        # x shape: [batch, channels, height, width] = [B, 768, 77, 43]
        
        # Upsample spatial dimensions
        x = self.upsample(x)  # Upsample to intermediate size [B, 768, 152, 224]

        # Adjust channel dimensions
        x = self.channel_adjust_conv(x)  # Reduce channels [B, 270, 152, 224]

        # Adaptive pool to target dimensions
        x = self.adaptive_pool(x)  # Final output size [B, 270, 152, 272]
        return x


class HRNetDINO2Adapter(nn.Module):
    def __init__(self, opt, target_shape=(768, 77, 43), device='cuda'):
        super().__init__()
        input_shape = (270, 152, 272)
        hidden_size = dinov2_models[opt.dinov2][1]
        self.input_channels, self.input_height, self.input_width = input_shape
        self.target_channels, self.target_height, self.target_width = target_shape
        
        # Convolution to increase channel depth to hidden_size
        self.channel_increase_conv = nn.Conv2d(self.input_channels, hidden_size, kernel_size=(1, 1), stride=1).to(device)

        # Downsampling layer to decrease resolution
        self.downsample = nn.Upsample(size=(77, 43), mode='bilinear', align_corners=False).to(device)  # Downsample to the target size

    def forward(self, x):
        # x shape: [batch, channels, height, width] = [B, 270, 152, 272]
        
        # Adjust channel dimensions
        x = self.channel_increase_conv(x)  # Increase channels [B, hidden_size, 152, 272]

        # Downsample spatial dimensions
        x = self.downsample(x)  # Downsample to target size [B, hidden_size, 77, 43]

        return x


class DistillationLoss(nn.Module):
    def __init__(self, device=0, loss_function='MSE'):
        super().__init__()
        self.device = device
        self.loss_function = loss_function
        # print(loss_function, '___________________________')
        if loss_function == 'MSE':
          self.loss = nn.MSELoss()
        elif loss_function == 'cosine':
          self.loss = nn.CosineEmbeddingLoss()
        else:
          assert "no valid loss function given"
          
  
    def forward(self, teacher_features, student_features):
        if self.loss_function == 'MSE':
          teacher_features, student_features = teacher_features.to(self.device), student_features.to(self.device)
          return self.loss(student_features, teacher_features).to(self.device)
        elif self.loss_function == 'cosine':
          teacher_features, student_features = teacher_features.to(self.device), student_features.to(self.device)
          input2_flat = student_features.view(student_features.size(0), -1)  # Shape: [batch_size, 270 * 152 * 272]
          input1_flat = teacher_features.view(teacher_features.size(0), -1)  # Shape: [batch_size, 270 * 152 * 272]
          return self.loss(input1_flat, input2_flat, target=torch.ones(teacher_features.shape[0]).to(self.device))
          
