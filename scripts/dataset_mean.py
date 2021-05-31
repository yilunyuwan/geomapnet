"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Computes the mean and std of pixels in a dataset
"""
import os.path as osp
import numpy as np
import argparse
import set_paths
from dataset_loaders.seven_scenes import SevenScenes
from dataset_loaders.tum import TUM
# from dataset_loaders.robotcar import RobotCar
from torchvision import transforms
from torch.utils.data import DataLoader
from common.train import safe_collate

parser = argparse.ArgumentParser(description='Dataset images statistics')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar', 'TUM'),
                    help='Dataset', required=True)
parser.add_argument('--scene', type=str, help='Scene name', required=True)
args = parser.parse_args()

data_dir = osp.join('..', 'data', args.dataset)
# crop_size_file = osp.join(data_dir, 'crop_size.txt')
# crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

data_transform = transforms.Compose([
  transforms.Resize(256),
  # transforms.RandomCrop(crop_size),
  transforms.ToTensor()])
depth_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.ToTensor() won't normalize int16 array to [0, 1]
    transforms.ToTensor(),
    # convenient for division operation
    # 15000 for fr1, 20000 for desk
	  # transforms.Lambda(lambda x: x.float()/20000.0)
	  # transforms.Lambda(lambda x: x.float()/65535.0)
	  transforms.Lambda(lambda x: x.float()/221.0)
  ])
# dataset loader
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=True, real=False,
  transform=data_transform)
if args.dataset == 'TUM':
  dset = TUM(mode=2, depth_transform=depth_transform, gt_path='associate_gt_fill_cmap.txt',**kwargs)
elif args.dataset == '7Scenes':
  dset = SevenScenes(**kwargs)
elif args.dataset == 'RobotCar':
  dset = RobotCar(**kwargs)
else:
  raise NotImplementedError

# accumulate
batch_size = 64
num_workers = batch_size
loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                    collate_fn=safe_collate)
# acc = np.zeros((3, crop_size[0], crop_size[1]))
# sq_acc = np.zeros((3, crop_size[0], crop_size[1]))
acc = np.zeros((3, 256, 341))
sq_acc = np.zeros((3, 256, 341))
# maximum, minimum = 0, 65535
for batch_idx, (imgs, _) in enumerate(loader):
  imgs = imgs['d'].numpy()
  acc += np.sum(imgs, axis=0)
  sq_acc += np.sum(imgs**2, axis=0)
  # print imgs.max(), imgs.min()
  '''
  maximum = imgs.max() if imgs.max() > maximum else maximum
  minimum = imgs.min() if imgs.min() < minimum else minimum

  mean_batch = np.mean(imgs)
  std_batch = np.std(imgs)
  print 'Mean = ', mean_batch, 'Std = ', std_batch
  '''
  if batch_idx % 50 == 0:
    print 'Accumulated {:d} / {:d}'.format(batch_idx*batch_size, len(dset))

N = len(dset) * acc.shape[1] * acc.shape[2]

# print maximum, minimum

mean_p = np.asarray([np.sum(acc[c]) for c in xrange(3)])
mean_p /= N
print 'Mean pixel = ', mean_p

# std = E[x^2] - E[x]^2
var_p = np.asarray([np.sum(sq_acc[c]) for c in xrange(3)])
var_p /= N
var_p -= (mean_p ** 2)
print 'Var. pixel = ', var_p

output_filename = osp.join('..', 'data', args.dataset, args.scene, 'depth_cmap_stats.txt')
np.savetxt(output_filename, np.vstack((mean_p, var_p)), fmt='%8.7f')
print '{:s} written'.format(output_filename)
