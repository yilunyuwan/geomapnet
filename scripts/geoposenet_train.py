"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Main training script for GeoMapNet
"""
import set_paths
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion, GeoPoseNetCriterion
from models.posenet import PoseNet, MapNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
import random


parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'geoposenet'),
  help='Model to train')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of absolute pose loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of relative pose loss')
parser.add_argument('--learn_recon', action='store_true',
  help='Learn the weight of photometric loss and ssim loss')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')
# power = optim_config.pop('power')
power = None

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = section.getfloat('sax')
saq = section.getfloat('saq')
if args.model.find('mapnet') >= 0 or args.model.find('geoposenet') >= 0:
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = section.getfloat('srx')
  srq = section.getfloat('srq')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print 'Using {:s} VO'.format(vo_lib)
if args.model.find('geoposenet') >= 0:
  slp = section.getfloat('slp')
  sls = section.getfloat('sls')
  # ld = section.getfloat('ld')
  # lp = section.getfloat('lp')
  # ls = section.getfloat('ls')

section = settings['training']
seed = section.getint('seed')
max_epoch = section.getint('n_epochs')

# perseve reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# model
feature_extractor = models.resnet34(pretrained=True)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model=='mapnet++'))
if args.model == 'geoposenet':
  model = MapNet(mapnet=posenet)
elif args.model == 'posenet':
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model == 'geoposenet':
  train_criterion = GeoPoseNetCriterion(sax=sax, saq=saq, srx=srx, srq=srq,slp=slp, sls=sls, learn_beta=args.learn_beta, learn_gamma=args.learn_gamma, learn_recon=args.learn_recon)#, ld=ld, lp=lp, ls=ls)
  val_criterion = GeoPoseNetCriterion()
elif args.model == 'posenet':
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
  if args.model.find('++') >= 0:
    kwargs = dict(kwargs, gps_mode=(vo_lib=='gps') )
    train_criterion = MapNetOnlineCriterion(**kwargs)
    val_criterion = MapNetOnlineCriterion()
  else:
    train_criterion = MapNetCriterion(**kwargs)
    val_criterion = MapNetCriterion()
else:
  raise NotImplementedError

# optimizer
param_list = [{'params': model.parameters()}]
# for poly decay learning rate adjustment policy
# fc_ids = list(map(id, model.mapnet.feature_extractor.fc.parameters()))
# fc_ids.extend(list(map(id, model.mapnet.fc_xyz.parameters())))
# fc_ids.extend(list(map(id, model.mapnet.fc_wpqr.parameters())))
# fc_params = filter(lambda p: id(p) in fc_ids, model.parameters()) 
# block_params = filter(lambda p: id(p) not in fc_ids, model.parameters()) 
# param_list = [{'params': fc_params},
#               {'params': block_params, 'lr': 4 * lr }]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
if args.learn_recon and hasattr(train_criterion, 'slp') and \
    hasattr(train_criterion, 'sls'):
  param_list.append({'params': [train_criterion.slp, train_criterion.sls]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, power=power, max_epoch=max_epoch, **optim_config)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

# transformers
tforms = [transforms.Resize(256)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  print 'Using ColorJitter data augmentation'
  tforms.append(transforms.ColorJitter(brightness=color_jitter,
    contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
depth_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
	  transforms.Lambda(lambda x: x.float())
  ])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
              target_transform=target_transform, seed=seed)
if args.model == 'geoposenet':
  if args.dataset == '7Scenes':
    kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip, depth_transform=depth_transform, mode=2)
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **kwargs)
  else:
    raise NotImplementedError
elif args.model == 'posenet':
  if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    train_set = SevenScenes(train=True, **kwargs)
    val_set = SevenScenes(train=False, **kwargs)
  elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    train_set = RobotCar(train=True, **kwargs)
    val_set = RobotCar(train=False, **kwargs)
  else:
    raise NotImplementedError
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  if args.model.find('++') >= 0:
    train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib=='gps'), **kwargs)
    val_set = None
  else:
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **kwargs)
else:
  raise NotImplementedError

# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
  args.model, config_name)
if args.learn_beta:
  experiment_name = '{:s}_learn_beta'.format(experiment_name)
if args.learn_gamma:
  experiment_name = '{:s}_learn_gamma'.format(experiment_name)
if args.learn_recon:
  experiment_name = '{:s}_learn_recon'.format(experiment_name)
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion)
lstm = args.model == 'vidloc'
geopose = args.model == 'geoposenet'
trainer.train_val(lstm=lstm, geopose=geopose)
