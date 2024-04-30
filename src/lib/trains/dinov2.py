from transformers import AutoImageProcessor, AutoModel
from transformers import Dinov2Config, Dinov2ForImageClassification

from PIL import Image
import requests
import torch


class Dinov2(Object):
  def __init__(self, opt):
    model_name = "facebook/dino-vitb8"
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, image_size=(opt.image_size))
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


class Knowledge_distillation(Object):

  def __init__(self, opt, model, optimizer=None):
    def __init__(
    self, opt, model, optimizer=None):

    self.opt = opt
    self.optimizer = optimizer
    self.student = model
    # self.loss_stats, self.loss = self._get_losses(opt)
    # self.model_with_loss = ModleWithLoss(model, self.loss)
    self.optimizer.add_param_group({'params': self.loss.parameters()})
    dino = Dinov2()
    self.teacher = dino(opt)
    
