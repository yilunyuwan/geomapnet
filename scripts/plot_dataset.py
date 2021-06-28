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
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar', 'TUM', 'AICL_NUIM'), help='Dataset', default='AICL_NUIM')
parser.add_argument('--scene', type=str, help='Scene name', default='livingroom')
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
stats_filename = osp.join(data_dir, args.scene, 'rgb_stats.txt')
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
train = not args.val
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
  transform=data_transform, target_transform=target_transform, seed=seed, gt_path='associate_gt.txt')

seq_names, prefix, suffix = None, None, None
if args.dataset == 'TUM':
  from dataset_loaders.tum import TUM
  if args.scene == 'fr1':
    seq_names = ['360', 'desk', 'desk2', 'room', 'floor']
    prefix = 'rgbd_dataset_freiburg1_'
  elif args.scene == 'human':
    seq_names = ['H1', 'H2', 'H3', 'H4', 'H5']
    suffix = '_pre_registereddata'
  elif args.scene == 'desk':
    seq_names = ['D1', 'D2', 'D3', 'D4', 'D5']
    suffix = '_pre_registereddata'
  elif args.scene == 'cabinet':
    seq_names = ['E1', 'E2', 'E3', 'E4', 'E5']
    suffix = '_pre_registereddata'
  else:
    raise NotImplementedError
elif args.dataset == 'AICL_NUIM':
  from dataset_loaders.tum import TUM
  if args.scene == 'livingroom':
    seq_names = ['livingroom1', 'livingroom2', 'living_room_traj0_frei_png', 'living_room_traj1_frei_png', 'living_room_traj2_frei_png', 'living_room_traj3_frei_png']
  elif args.scene == 'office':
    seq_names = ['office1', 'office2', 'office_traj0_frei_png', 'office_traj1_frei_png', 'office_traj2_frei_png', 'office_traj3_frei_png']
  else:
    raise NotImplementedError
elif args.dataset == '7Scenes':
  from dataset_loaders.seven_scenes import SevenScenes
  seq_nums = {'chess': 6, 'fire': 4, 'heads': 2, 'office': 10, 'pumpkin': 8, 'redkitchen': 14, 'stairs': 6}
  if args.scene in seq_nums:
    seq_num = seq_nums[args.scene]
  else:
    raise NotImplementedError
  seq_names = [i for i in xrange(1, seq_num+1)]
else:
  raise NotImplementedError

data_sets = []
Ls = []
del_names = []
for name in seq_names:
  seq_path = name
  if prefix is not None:
    seq_path = prefix + seq_path
  if suffix is not None:
    seq_path = seq_path + suffix
  seq_kwargs = dict(kwargs, draw_seq=seq_path)
 
  if args.dataset == 'TUM' or args.dataset == 'AICL_NUIM':
    data_set = TUM(**seq_kwargs)
  elif args.dataset == '7Scenes':
    try:
      data_set = SevenScenes(**seq_kwargs)
    except OSError as e:
      print '[Never Mind]{:s} seq-{:02d} does not exist, IOError: {:s}'.format(args.scene, seq_path, e)
      data_set = None
      del_names.append(name)
  else:
    raise NotImplementedError

  if data_set:
    L = len(data_set)
    data_sets.append(data_set)
    Ls.append(L)

for name in del_names:
  seq_names.remove(name)

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1

loaders = []
for data_set in data_sets:
  loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                  num_workers=5, pin_memory=True)
  loaders.append(loader)

                    
# # activate GPUs
# CUDA = torch.cuda.is_available()
# torch.manual_seed(seed)
# if CUDA:
#   torch.cuda.manual_seed(seed)
#   model.cuda()

'''
pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses
'''
# create figure object
fig = plt.figure()
if args.dataset != '7Scenes' and args.dataset != 'TUM' and args.dataset != 'AICL_NUIM':
  ax = fig.add_subplot(111)
else:
  ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
# different colors:
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightcoral', 'orangered','chocolate', 'orange', 'yellow', 'lightgreen', 'deepskyblue']

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

'''
# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                       targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                       targ_poses[:, 3:])])
#eval_func = np.mean if args.dataset == 'RobotCar' else np.median
#eval_str  = 'Mean' if args.dataset == 'RobotCar' else 'Median'
#t_loss = eval_func(t_loss)
#q_loss = eval_func(q_loss)
#print '{:s} error in translation = {:3.2f} m\n' \
#      '{:s} error in rotation    = {:3.2f} degrees'.format(eval_str, t_loss,
print 'Error in translation: median {:4.3f} m,  mean {:4.3f} m\n' \
    'Error in rotation: median {:4.3f} degrees, mean {:4.3f} degree'.format(np.median(t_loss), np.mean(t_loss),
                    np.median(q_loss), np.mean(q_loss))
'''


if DISPLAY:
  # plt.show(block=False)
  plt.show(block=True)

if args.output_dir is not None:
  scene_name = '{:s}_{:s}'.format(args.dataset, args.scene)
  image_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.png'.format(scene_name))
  fig.savefig(image_filename)
  print '{:s} saved'.format(image_filename)
