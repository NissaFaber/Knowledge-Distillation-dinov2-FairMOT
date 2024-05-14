

import torch
from torch import nn
from transformers import ViTImageProcessor, Dinov2Model, Dinov2Config, AutoBackbone
from torch.functional import F



dinov2_models = {'base': 'facebook/dinov2-base' ,
                 'small': 'facebook/dinov2-small',
                 'large' : 'facebook/dinov2-large',
                 'giant' : 'facebook/dinov2-giant'}




class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        self.device = torch.device(opt.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(dinov2_models[opt.dinov2], size={'height': 1088, 'width': 608}, do_rescale=False, reshape_hidden_states=True)
        config = Dinov2Config.from_pretrained(dinov2_models[opt.dinov2], reshape_hidden_states=True, output_hidden_states=True)
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
    def __init__(self, hidden_size=384, target_shape=(270, 152, 272), device='0'):
        super().__init__()
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



class DistillationLoss(nn.Module):
    def __init__(self, device=0):
        super().__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features, student_features):
        teacher_features, student_features = teacher_features.to(self.device), student_features.to(self.device)
        return self.mse_loss(student_features, teacher_features)

# opt = type('opt', (object,), {'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
# distillation_loss = DistillationLoss(device=opt.device)
# adapter = DINO2HRNetAdapter(device=opt.device)
# dinov2 = Dinov2(opt)
