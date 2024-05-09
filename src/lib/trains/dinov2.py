import torch
from torch import nn
from transformers import ViTImageProcessor, Dinov2Model, Dinov2Config

class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        config = Dinov2Config(
            reshape_hidden_states=True,
            output_hidden_states=True
        )
        model_name = "facebook/dino-vitb8"
        self.device = torch.device(opt.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.model = Dinov2Model(config)
        self.model.eval()

    def forward(self, batch_images):
        # Check if images need rescaling or not, and adjust preprocessing accordingly
        
        inputs = self.feature_extractor(images=batch_images['input'], return_tensors="pt", padding=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        return features

class DINO2HRNetAdapter(nn.Module):
    def __init__(self, hidden_size=768, target_shape=(270, 152, 272), device='cpu'):
        super().__init__()
        self.target_channels, self.target_height, self.target_width = target_shape
        
        # Create convolution and adaptive pooling layers on the specified device
        self.reshape_conv = nn.Conv2d(hidden_size, self.target_channels, 1).to(device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_height, self.target_width)).to(device)
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        # Drop class token if present (assuming it's the first token)
        x = x[:, 1:, :]  # Ensure x is already on the correct device, handled outside
        
        # Reshape and pass through the convolution and pooling
        batch_size, seq_len, hidden_size = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, hidden_size, int(seq_len**0.5), int(seq_len**0.5))
        x = self.reshape_conv(x)
        x = self.adaptive_pool(x)
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
