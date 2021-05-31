"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np

import os
os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

import sys
sys.path.insert(0, '../')

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

class PoseNet_S1(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0):
    super(PoseNet_S1, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features

    if self.two_stream_mode == 0:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = feature_extractor
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, 3)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, 3)
    elif self.two_stream_mode == 1:
      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = feature_extractor
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, 3)
      self.depth_fc_wpqr = nn.Linear(feat_dim, 3)
    else:
      print 'two_stream_mode shoud be 0 or 1 at the first stage'
      raise NotImplementedError

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      if self.two_stream_mode == 0:
        init_modules = [self.rgb_feature_extractor.fc, self.rgb_fc_xyz, self.rgb_fc_wpqr]
      else:
        init_modules = [self.depth_feature_extractor.fc, self.depth_fc_xyz, self.depth_fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, data):
    if self.two_stream_mode == 0:
      # RGB feature extractor
      x = self.rgb_feature_extractor(data)
      x = F.relu(x)
      xyz  = self.rgb_fc_xyz(x)
      log_q = self.rgb_fc_wpqr(x)
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      xyz  = self.depth_fc_xyz(x)
      log_q = self.depth_fc_wpqr(x)
    return torch.cat((xyz, log_q), dim=1)

class PoseNet_S2(nn.Module):
  def __init__(self, rgb_f_e, depth_f_e, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False):
    super(PoseNet_S2, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    fe_out_planes = rgb_f_e.fc.in_features

    # RGB feature extractor
    # replace the last FC layer in feature extractor
    self.rgb_feature_extractor = rgb_f_e
    self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
    # Depth feature extractor
    # replace the last FC layer in feature extractor
    self.depth_feature_extractor = depth_f_e
    self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
    '''
    self.fc_rgbd_fusion = nn.Linear(2*feat_dim, feat_dim)
    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 3)
    '''
    self.fc_xyz  = nn.Linear(2*feat_dim, 3)
    self.fc_wpqr = nn.Linear(2*feat_dim, 3)

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.fc_xyz, self.fc_wpqr]#, self.fc_rgbd_fusion]

    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, rgbs, depths):
    # RGB feature extractor
    x = self.rgb_feature_extractor(rgbs)
    x = F.relu(x)
    # Depth feature extractor
    y = self.depth_feature_extractor(depths)
    y = F.relu(y)
    # concatenate x and y
    x = torch.cat((x, y), dim=1)
    # x = self.fc_rgbd_fusion(x)
    xyz  = self.fc_xyz(x)
    log_q = self.fc_wpqr(x)

    return torch.cat((xyz, log_q), dim=1)

class MapNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, mapnet, two_stream_mode=0):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(MapNet, self).__init__()
    self.mapnet = mapnet
    self.two_stream_mode = two_stream_mode

  def forward(self, data, depths=None):
    """
    :param data: image blob (N x T x C x H x W) color images when mode=0, dn images when mode=1
    :param depths: image blob (N x T x C x H x W) depths is not None when mode>=2
    :return: pose nnd nd
     (N x T x 6)
    """
    s = data.size()
    data = data.view(-1, *s[2:])
    if self.two_stream_mode >= 2:
      if depths is None:
        raise Exception('Depth image cannot be None when using two stream CNN')
      depths = depths.view(-1, *s[2:])
      poses = self.mapnet(data, depths)
    else:
      poses = self.mapnet(data)
    
    poses = poses.view(s[0], s[1], -1)
    return poses
