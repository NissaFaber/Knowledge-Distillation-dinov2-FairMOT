from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from .dinov2 import Dinov2, DINO2HRNetAdapter, DistillationLoss


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs, embeddings = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], embeddings, loss, loss_stats


class dinoModelloss(torch.nn.Module):
  def __init__(self, model, opt):
    super(dinoModelloss, self).__init__()
    self.model = model
    self.loss = DistillationLoss(device = opt.device)
    self.adapter = DINO2HRNetAdapter(device = opt.device)

  def forward(self, batch, embeddings):
    outputs = self.model(batch)
    hrnet_compatible_outputs = self.adapter(outputs)
    loss = self.loss(hrnet_compatible_outputs, embeddings)
    return loss
    



class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.teacher = Dinov2(opt = opt)
    self.optimizer = optimizer
    self.alpha = opt.alpha
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.teacher_with_loss = dinoModelloss(self.teacher, opt)
    self.optimizer.add_param_group({'params': self.loss.parameters()})
    self.optimizer.add_param_group({'params': self.teacher_with_loss.adapter.parameters()})


  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.teacher = self.teacher.to(device)
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    teacher_with_loss = self.teacher_with_loss
    if phase == 'train':
      model_with_loss.train()
      #teacher already in eval() mode
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      # print(dinoModelloss.DINO2HRNetAdapter.Conv2d.weight)
      output, embeddings, loss, loss_stats = model_with_loss(batch)
      loss_teacher = teacher_with_loss(batch, embeddings)

      # print(loss_teacher,'check')
      
      # print(embeddings, embeddings.shape,'student____________________________________________\n', output_teacher, output_teacher.shape, 'teacher_____________________________________________' )
      # adapter = DINO2HRNetAdapter(device = opt.device)
      # hrnet_compatible_outputs = adapter(output_teacher).to(opt.device)
      # print(hrnet_compatible_outputs.shape)
      
      # distillation_loss = DistillationLoss(device = opt.device)
      # lossTest = distillation_loss(hrnet_compatible_outputs, embeddings)
      # print(lossTest, 'distillation loss')
      
      loss = loss.mean() + self.alpha * loss_teacher
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {} {:.4f} '.format(loss_teacher, l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats, batch
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
