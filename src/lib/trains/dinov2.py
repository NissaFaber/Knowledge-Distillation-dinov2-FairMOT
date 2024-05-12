import torch
from torch import nn
from transformers import ViTImageProcessor, Dinov2Model, Dinov2Config


dinov2_models = {'base': 'facebook/dinov2-base' ,
                 'small': 'facebook/dinov2-small',
                 'large' : 'facebook/dinov2-large',
                 'giant' : 'facebook/dinov2-giant'}




class Dinov2(nn.Module):
    def __init__(self, opt):
        super(Dinov2, self).__init__()
        config = Dinov2Config(
            reshape_hidden_states=True,
            output_hidden_states=True
        )
        model_name = dinov2_models[opt.dinov2]
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







# import torch
# from torch import nn
# from transformers import ViTImageProcessor, Dinov2Model, Dinov2Config, AutoModel


# dinov2_models = {'base': 'facebook/dinov2-base' ,
#                  'small': 'facebook/dinov2-small',
#                  'large' : 'facebook/dinov2-large',
#                  'giant' : 'facebook/dinov2-giant'}




# class Dinov2(nn.Module):
#     def __init__(self, opt):
#         super(Dinov2, self).__init__()
#         model_name = dinov2_models[opt.dinov2]      
#         self.device = torch.device(opt.device)
#         self.feature_extractor = ViTImageProcessor.from_pretrained('facebook/dinov2-small', size={'height': 1088, 'width': 608}, do_rescale=False)
#         self.model = Dinov2Model.from_pretrained('facebook/dinov2-small')        
#         self.model.eval()
#         self.depatch = Depatchify()

#     def forward(self, batch_images):
#         # Check if images need rescaling or not, and adjust preprocessing accordingly
        
#         inputs = self.feature_extractor(images=batch_images['input'], return_tensors="pt", padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         print(inputs['pixel_values'].shape)
#         outputs = self.model(**inputs)
#         features = outputs.last_hidden_state
#         print(self.depatch(features), ' depathcified')

#         return features


# class Depatchify(torch.nn.Module):
#     def __init__(self, patch_size=16, orig_dims=(1088, 608), embed_dim=384):
#         super().__init__()
#         self.patch_size = patch_size
#         self.orig_height, self.orig_width = orig_dims
#         self.embed_dim = embed_dim


#     def forward(self, x):
#         batch_size, num_patches, _ = x.shape

#         # Calculate the number of patches along height and width
#         num_patches_height = self.orig_height // self.patch_size
#         num_patches_width = self.orig_width // self.patch_size
        
#         # Reshape back to (batch_size, channels, height_patches, width_patches)
#         x = x.view(batch_size, num_patches_height, num_patches_width, self.patch_size, self.patch_size, self.embed_dim)
        
#         # Permute to bring patches into a continuous image layout
#         x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        
#         # Flatten the patch dimensions into full image dimensions
#         x = x.view(batch_size, self.embed_dim, self.orig_height, self.orig_width)

#         return x

# # # Assuming your embeddings are from a model that used 16x16 patches
# # depatchifier = Depatchify(patch_size=16, orig_dims=(1088, 608), embed_dim=384)
# # embeddings = torch.randn(2, 3312, 384)  # Simulated embeddings
# # reconstructed_images = depatchifier(embeddings)
# # print(reconstructed_images.shape)  # Should show torch.Size([2, 384, 1088, 608])


# class DINO2HRNetAdapter(nn.Module):
#     def __init__(self, hidden_size=384, target_shape=(270, 152, 272), device='cpu'):
#         super().__init__()
#         self.target_channels, self.target_height, self.target_width = target_shape
#         # Adjust the size using convolution that changes channels and a transitional shape that approximates the target shape
#         self.approx_sqrt = int((3311)**0.5)  # Approximate sqrt to get a nearly square shape
#         self.transition_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 1), stride=1).to(device)  # Maintain the same channel depth
#         self.reshape_conv = nn.Conv2d(hidden_size, self.target_channels, (1, 1)).to(device)  # Reduce the channel depth to target channels
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((self.target_height, self.target_width)).to(device)  # Pool to the exact target dimensions
        
#     def forward(self, x):
#         # Assume x shape: [batch, seq_len, hidden_size] with seq_len = 3311 (after removing class token)
#         batch_size, seq_len, hidden_size = x.shape
#         x = x[:, 1:, :]  # Drop class token
        
#         # First, transition reshape to the approximate square form
#         x = x.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
#         x = x.view(batch_size, hidden_size, self.approx_sqrt, self.approx_sqrt)  # Reshape to approx square - adjust size as close as possible to square
        
#         # Pass through transitional convolution to maintain the depth
#         x = self.transition_conv(x)
        
#         # Now apply the reshaping convolution to adjust channels
#         x = self.reshape_conv(x)
        
#         # Finally, adaptive pooling to target dimensions
#         x = self.adaptive_pool(x)
#         return x



# class DistillationLoss(nn.Module):
#     def __init__(self, device=0):
#         super().__init__()
#         self.device = device
#         self.mse_loss = nn.MSELoss()

#     def forward(self, teacher_features, student_features):
#         teacher_features, student_features = teacher_features.to(self.device), student_features.to(self.device)
#         return self.mse_loss(student_features, teacher_features)

# # opt = type('opt', (object,), {'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
# # distillation_loss = DistillationLoss(device=opt.device)
# # adapter = DINO2HRNetAdapter(device=opt.device)
# # dinov2 = Dinov2(opt)

