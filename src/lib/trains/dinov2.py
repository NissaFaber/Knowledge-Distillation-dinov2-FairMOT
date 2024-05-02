import torch
from torch import nn
from transformers import AutoFeatureExtractor, Dinov2Model, Dinov2Config

class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1, device='cpu'):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.device = device
        self.classifier = nn.Conv2d(in_channels, num_labels, (1,1)).to(self.device)

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2).to(self.device)
        return self.classifier(embeddings)

class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        config = Dinov2Config(
            reshape_hidden_states=True,
            output_hidden_states=True
        )
        model_name = "facebook/dino-vitb8"
        self.device = torch.device(opt.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name).to(self.device)
        self.model = Dinov2Model(config).to(self.device)
        self.model.eval()

    def forward(self, batch_images):
        inputs = self.feature_extractor(images=batch_images['input'], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        return features

class DINO2HRNetAdapter(nn.Module):
    def __init__(self, hidden_size=768, target_shape=(270, 152, 272), device):
        super().__init__()
        self.device = device
        self.target_channels, self.target_height, self.target_width = target_shape
        self.reshape_conv = nn.Conv2d(hidden_size, self.target_channels, 1).to(self.device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_height, self.target_width))

    def forward(self, x):
        x = x[:, 1:, :].to(self.device)
        batch_size, seq_len, hidden_size = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, hidden_size, int(seq_len**0.5), int(seq_len**0.5))
        x = self.reshape_conv(x)
        x = self.adaptive_pool(x)
        return x

class DistillationLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features, student_features):
        teacher_features, student_features = teacher_features.to(self.device), student_features.to(self.device)
        return self.mse_loss(student_features, teacher_features)

Usage example assuming opt.device is set
# opt = type('opt', (object,), {'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
# distillation_loss = DistillationLoss(device=opt.device)
# adapter = DINO2HRNetAdapter(device=opt.device)
# dinov2 = Dinov2(opt)
