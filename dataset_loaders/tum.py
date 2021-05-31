"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
pytorch data loader for the TUM dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from utils import load_image, load_depth
import sys
import pickle

sys.path.insert(0, '../')
from common.pose_utils import process_poses4tum

class TUM(data.Dataset):
    def __init__(self, scene, data_path, train, 
    transform=None, depth_transform=None, dn_transform=None,target_transform=None, mode=0, seed=7, real=False, skip_images=False, gt_path='associate_gt_fill.txt'):
      """
      :param scene: scene name ['fr1', 'fr2']
      :param data_path: root TUM data directory.
      Usually '../data/deepslam_data/TUM'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the color images
      :param depth_transform: depth_transform to apply to the color images
      :param target_transform: transform to apply to the poses
      :param mode: 0: just color image {'c': c_img}, 1: just depth image {'d': d_img}, 2: {'c': c_img, 'd': d_img}, 2 and dn_transform:  {'c': c_img, 'd': d_img, 'dn': depth used to be trained(standardization d_img)}
      :param real: If True, load poses from SLAM/integration of VO
      :param skip_images: If True, skip loading images and return None instead
      """
      self.mode = mode
      self.transform = transform
      self.depth_transform = depth_transform
      self.dn_transform = dn_transform
      self.target_transform = target_transform
      self.skip_images = skip_images
      np.random.seed(seed)

      # directories
      base_dir = osp.join(osp.expanduser(data_path), scene)
      data_dir = osp.join('..', 'data', 'TUM', scene)

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      with open(split_file, 'r') as f:
        seqs = [l.splitlines()[0] for l in f if not l.startswith('#')]

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      gt_offset = int(0)
      for seq in seqs:
        seq_dir = osp.join(base_dir, seq)
        seq_data_dir = osp.join(data_dir, seq)
        # associate_gt_path = osp.join(seq_dir, 'associate_gt.txt')
        associate_gt_path = osp.join(seq_dir, gt_path)
        rgbs = np.loadtxt(associate_gt_path, dtype=str, usecols=1)
        depths = np.loadtxt(associate_gt_path, dtype=str, usecols=3)
        pss = np.loadtxt(associate_gt_path, usecols=(5, 6, 7, 11, 8, 9, 10))
        ps[seq] = np.asarray(pss)

        frame_idx = np.array(xrange(len(rgbs)), dtype=np.int)
        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(rgbs)
        c_imgs = [osp.join(seq_dir, rgb) for rgb in rgbs]
        d_imgs = [osp.join(seq_dir, depth) for depth in depths]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)


      # convert pose to translation + log quaternion
      self.poses = np.empty((0, 6))
      for seq in seqs:
        pss = process_poses4tum(ps[seq])
        self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
      if self.skip_images:
        img = None
        pose = self.poses[index]
      else:
        if self.mode == 0:
          img = None
          while img is None:
            img = {'c': load_image(self.c_imgs[index])}
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 1:
          img = None
          while img is None:
            img = {'d': load_depth(self.d_imgs[index])}
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 2:
          c_img = None
          d_img = None
          while (c_img is None) or (d_img is None):
            c_img = load_image(self.c_imgs[index])
            # load_image use pil_loader which convert d_image into RGB (3 channels, 8 bits per channel), while d_img is 16 bits png image which should be convert into I (32 bits, 1 channel)
            d_img = load_depth(self.d_imgs[index])
            # d_img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          img = {'c': c_img, 'd': d_img}
          index -= 1
        else:
          raise Exception('Wrong mode {:d}'.format(self.mode))

      if self.target_transform is not None:
        pose = self.target_transform(pose)

      if self.skip_images:
        return img, pose

      if self.mode >= 2 and self.transform is not None and self.depth_transform is not None and self.dn_transform is not None:
        img = {'c': self.transform(img['c']), 'd': self.depth_transform(img['d']), 'dn': self.dn_transform(img['d'])}
      elif self.mode == 1 and self.dn_transform is not None:
        img = {'dn': self.dn_transform(img['d'])}
      elif self.mode == 0 and self.transform is not None:
        img = {'c': self.transform(img['c'])}
      else:
        raise Exception('Missing transform for mode {:d}'.format(self.mode))
      return img, pose

    def __len__(self):
      return self.poses.shape[0]

def main():
  """
  visualizes the dataset
  """
  from common.vis_utils import show_batch, show_stereo_batch, show_depth_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  import torch
  import numpy as np

  seq = 'desk'
  mode = 2
  num_workers = 0
  data_dir = osp.join('..', 'data', 'TUM')
  stats_file = osp.join(data_dir, seq, 'stats.txt')
  stats = np.loadtxt(stats_file)
  depth_stats = np.loadtxt( osp.join(data_dir, seq, 'depth_stats.txt') )
  print stats
  print depth_stats
  transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float()),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  dn_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.ToTensor() won't normalize int16 array to [0, 1]
    transforms.ToTensor(),
    # from [B, H, W] to [B, C, H, W] and normalization
	  transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0).float() / 65535),
    transforms.Normalize(mean=depth_stats[0], std=np.sqrt(depth_stats[1]))
  ])
  dset = TUM(seq, '../data/deepslam_data/TUM', train=False, transform=transform,
    depth_transform=transform, mode=mode, dn_transform=dn_transform)
  print 'Loaded TUM sequence {:s}, length = {:d}'.format(seq,
    len(dset))

  data_loader = data.DataLoader(dset, batch_size=5, shuffle=True,
    num_workers=num_workers)

  batch_count = 0
  N = 2
  for (imgs, poses) in data_loader:
    print 'Minibatch {:d}'.format(batch_count)
    if mode == 0:
      show_batch(make_grid(imgs['c'], nrow=1, padding=25))
    elif mode == 1:
      show_depth_batch(make_grid(imgs['d'], nrow=1, padding=25))
    elif mode == 2:
      color_show = (imgs['c'] + 1.0) / 2.0
      # color_show = imgs['c']
      lb = make_grid(color_show, nrow=1, padding=25)
      rb = make_grid(imgs['d'].float()/65535.0, nrow=1, padding=25)
      print imgs['dn'].shape
      print torch.max(imgs['c']), torch.min(imgs['c']), torch.mean(imgs['c'])
      print imgs['c'].dtype
      show_stereo_batch(lb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
