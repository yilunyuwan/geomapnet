"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import set_paths
from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
  calc_vos_safe_fc, calc_vos_safe
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
import matplotlib
DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
  matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, models
import cPickle

# config
parser = argparse.ArgumentParser(description='Evaluation script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar', 'TUM'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++'),
  help='Model to use (mapnet includes both MapNet and MapNet++ since their'
       'evluation process is the same and they only differ in the input weights'
       'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--output_dir', type=str, default=None,
  help='Output image directory')
parser.add_argument('--pose_graph', action='store_true',
  help='Turn on Pose Graph Optimization')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

seed = 7
# settings = configparser.ConfigParser()
# with open(args.config_file, 'r') as f:
#   settings.read_file(f)
# seed = settings.getint('training', 'seed')
# section = settings['hyperparameters']
# dropout = section.getfloat('dropout')
# if (args.model.find('mapnet') >= 0) or args.pose_graph:
#   steps = section.getint('steps')
#   skip = section.getint('skip')
#   real = section.getboolean('real')
#   variable_skip = section.getboolean('variable_skip')
#   fc_vos = args.dataset == 'RobotCar'
#   if args.pose_graph:
#     vo_lib = section.get('vo_lib')
#     sax = section.getfloat('s_abs_trans', 1)
#     saq = section.getfloat('s_abs_rot', 1)
#     srx = section.getfloat('s_rel_trans', 20)
#     srq = section.getfloat('s_rel_rot', 20)

# # model
# feature_extractor = models.resnet34(pretrained=False)
# posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
# if (args.model.find('mapnet') >= 0) or args.pose_graph:
#   model = MapNet(mapnet=posenet)
# else: # geoposenet use posenet model when evaluation
#   model = posenet
# model.eval()

# # loss functions
# t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
# q_criterion = quaternion_angular_error

# # load weights
# weights_filename = osp.expanduser(args.weights)
# if osp.isfile(weights_filename):
#   loc_func = lambda storage, loc: storage
#   checkpoint = torch.load(weights_filename, map_location=loc_func)
#   load_state_dict(model, checkpoint['model_state_dict'])
#   print 'Loaded weights from {:s}'.format(weights_filename)
# else:
#   print 'Could not load weights from {:s}'.format(weights_filename)
#   sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_filename)
# transformer
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# # read mean and stdev for un-normalizing predictions
# pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
# pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# dataset
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
train_kwargs = dict(scene=args.scene, data_path=data_dir, train=True,
  transform=data_transform, target_transform=target_transform, seed=seed)
test_kwargs = dict(scene=args.scene, data_path=data_dir, train=False,
  transform=data_transform, target_transform=target_transform, seed=seed)

if args.dataset == 'TUM':
  from dataset_loaders.tum import TUM
  train_data_set = TUM(**train_kwargs)
  test_data_set = TUM(**test_kwargs)
elif args.dataset == '7Scenes':
  from dataset_loaders.seven_scenes import SevenScenes
  train_data_set = SevenScenes(**train_kwargs)
  test_data_set = SevenScenes(**test_kwargs)
else:
  raise NotImplementedError

data_sets = [train_data_set, test_data_set]
Ls = [len(train_data_set), len(test_data_set)]
# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1

loaders = []
for data_set in data_sets:
  loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                  num_workers=5, pin_memory=True)
  loaders.append(loader)

                    

# create figure object
fig = plt.figure()
if args.dataset != '7Scenes' and args.dataset != 'TUM':
  ax = fig.add_subplot(111)
else:
  ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
# different colors:
colors = ['g', 'r']
seq_names = ['train', 'test']
# inference loop

for i, (data_set, L, loader) in enumerate(zip(data_sets, Ls, loaders)):
  targ_poses = np.zeros((L, 7))  # store all target poses
  for batch_idx, (data, target) in enumerate(loader):
    if batch_idx % 200 == 0:
      print 'Image {:d} / {:d}'.format(batch_idx, len(loader))
    '''
    # output : 1 x 6 or 1 x STEPS x 6
    _, output = step_feedfwd(data['c'], model, CUDA, train=False)
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    '''
    target = target.numpy().reshape((-1, target.size()[-1]))

    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))
    targ_poses[batch_idx, :] = target

  # plot on the figure object
  # ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
  ss = 1
  # draw connecting line 
  x = targ_poses[::ss, 0].T
  y = targ_poses[::ss, 1].T
  z = targ_poses[::ss, 2].T
  ax.plot(x, y, z, c=colors[i], label=seq_names[i])

ax.set_title('{:s}'.format(args.scene))
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.legend()
ax.view_init(azim=119, elev=13)


if DISPLAY:
  # plt.show(block=False)
  plt.show(block=True)

if args.output_dir is not None:
  scene_name = '{:s}_{:s}_train_test'.format(args.dataset, args.scene)
  image_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.png'.format(scene_name))
  fig.savefig(image_filename)
  print '{:s} saved'.format(image_filename)
