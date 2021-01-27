"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import torch.optim as optim

class Optimizer:
  """
  Wrapper around torch.optim + learning rate
  """
  def __init__(self, params, method, base_lr, weight_decay, power, max_epoch, **kwargs):
    self.method = method
    self.base_lr = base_lr
    self.power = power
    self.max_epoch = max_epoch

    if self.method == 'sgd':
      self.lr_decay = kwargs.pop('lr_decay')
      self.lr_stepvalues = sorted(kwargs.pop('lr_stepvalues'))
      self.learner = optim.SGD(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'adam':
      self.learner = optim.Adam(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'rmsprop':
      self.learner = optim.RMSprop(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)

              
  def poly_adjust_lr(self, i_epoch):
      lr = self.base_lr * ((1 - float(i_epoch) / self.max_epoch)**(self.power))
      self.learner.param_groups[0]['lr'] = lr
      self.learner.param_groups[1]['lr'] = 4 * lr
      return lr

  def adjust_lr(self, epoch):
    if self.method != 'sgd':
      return self.base_lr

    decay_factor = 1
    for s in self.lr_stepvalues:
      if epoch < s:
        break
      decay_factor *= self.lr_decay

    lr = self.base_lr * decay_factor

    for param_group in self.learner.param_groups:
      param_group['lr'] = lr

    return lr

  def mult_lr(self, f):
    for param_group in self.learner.param_groups:
      param_group['lr'] *= f

def main():
  """
  test polynomial decay of Adam
  """

  from torchvision import  models
  from criterion import GeoPoseNetCriterion
  import configparser
  import json
  import sys
  sys.path.insert(0, '../')
  from models.posenet import PoseNet, MapNet


  config_file = '../scripts/configs/geoposenet.ini'
  settings = configparser.ConfigParser()
  with open(config_file, 'r') as f:
    settings.read_file(f)
  section = settings['optimization']
  optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
  opt_method = section['opt']
  lr = optim_config.pop('lr')
  weight_decay = optim_config.pop('weight_decay')
  power = optim_config.pop('power')

  feature_extractor = models.resnet50(pretrained=True)
  posenet = PoseNet(feature_extractor, pretrained=True)
  model = MapNet(mapnet=posenet)
  model.train()
  model.cuda()

  fc_ids = list(map(id, model.mapnet.feature_extractor.fc.parameters()))
  fc_ids.extend(list(map(id, model.mapnet.fc_xyz.parameters())))
  fc_ids.extend(list(map(id, model.mapnet.fc_wpqr.parameters())))
  fc_params = filter(lambda p: id(p) in fc_ids, model.parameters()) 
  block_params = filter(lambda p: id(p) not in fc_ids, model.parameters()) 
  param_list = [{'params': fc_params},
                {'params': block_params, 'lr': 4 * lr }]

  optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, **optim_config)

  max_epoch = 50
  for i in range(max_epoch):
    optimizer.poly_adjust_lr(base_lr=lr, i_epoch=i, max_epoch=max_epoch, power=power)
    optimizer.learner.zero_grad()
    # loss.backward()
    optimizer.learner.step()
    print i, optimizer.learner.param_groups[0]['lr'], optimizer.learner.param_groups[1]['lr']


if __name__ == '__main__':
    main()