"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
This module implements the various loss functions (a.k.a. criterions) used
in the paper
"""

from pose_utils import calc_vo_logq2q
import pose_utils
from reconstruction_utils import reconstruction
import torch
from torch import nn
from ssim import ssim, ms_ssim

class QuaternionLoss(nn.Module):
  """
  Implements distance between quaternions as mentioned in
  D. Huynh. Metrics for 3D rotations: Comparison and analysis
  """
  def __init__(self):
    super(QuaternionLoss, self).__init__()

  def forward(self, q1, q2):
    """
    :param q1: N x 4
    :param q2: N x 4
    :return: 
    """
    loss = 1 - torch.pow(pose_utils.vdot(q1, q2), 2)
    loss = torch.mean(loss)
    return loss

class PoseNetCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
      saq=0.0, learn_beta=False):
    super(PoseNetCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

  def forward(self, pred, targ):
    """
    :param pred: N x 7
    :param targ: N x 7
    :return: 
    """
    abs_t_loss = self.t_loss_fn(pred[:, :3], targ[:, :3])
    abs_q_loss = self.q_loss_fn(pred[:, 3:], targ[:, 3:])

    loss = torch.exp(-self.sax) * abs_t_loss + \
      self.sax +\
     torch.exp(-self.saq) * abs_q_loss +\
      self.saq
    return loss, abs_t_loss, abs_q_loss, 0, 0

"""
modified from MapNetCriterion
"""
class GeoPoseNetCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False, ld=1, lp=0.01, ls=0.1):
    """
    Implements L_D from eq. 2 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    """
    super(GeoPoseNetCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.ld = ld
    self.lp = lp
    self.ls = ls
    """ without consideration of relative pose error for the time being
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)
    """

  def forward(self, pred, targ, color, depth):
    """
    :param pred: N x T x 6
    :param targ: N x T x 6
    :param color: N x T x 3 x H x W 
    :param depth: N x T x 1 x H x W
    :return:
    """

    # absolute pose loss
    s = pred.size()
    '''
    abs_loss =\
      torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],
                                            targ.view(-1, *s[2:])[:, :3]) + \
      self.sax + \
      torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                            targ.view(-1, *s[2:])[:, 3:]) + \
      self.saq
    '''
    t_loss = self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3])
    q_loss = self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                            targ.view(-1, *s[2:])[:, 3:])
    abs_loss =  t_loss - self.saq * q_loss
    reconstruction_loss, ms_ssim_loss = torch.Tensor([0]).type_as(t_loss), torch.Tensor([0]).type_as(t_loss)
    '''
    # get the photometric reconstruction
    # u_{src} = K T_{tgt->src} D_{tgt} K^{-1} u_{tgt}
    mid = s[1] / 2
    # src_imgs, tgt_imgs: (N*ceil(T/2)) x 3 x H x W 
    # tgt_depths: (N*ceil(T/2)) x 1 x H x W 
    src_imgs = color[:, :mid+1, ...].reshape(-1, *color.size()[2:])
    tgt_imgs = color[:, mid:, ...].reshape(-1, *color.size()[2:])
    tgt_depths = depth[:, mid:, ...].reshape(-1, *depth.size()[2:])
    
    src_pred = pred[:, :mid+1, ...].reshape(-1, *s[2:])
    tgt_pred = pred[:, mid:, ...].reshape(-1, *s[2:])
    src_targ = targ[:, :mid+1, ...].reshape(-1, *s[2:])
    tgt_targ = targ[:, mid:, ...].reshape(-1, *s[2:])
    # pred_relative_poses, targ_relative_poses: (N*ceil(T/2)) x 7
    pred_relative_poses = calc_vo_logq2q(src_pred, tgt_pred)
    targ_relative_poses = calc_vo_logq2q(src_targ, tgt_targ) 
    
    projected_imgs, valid_points = reconstruction(src_imgs, tgt_depths, pred_relative_poses)
    diff = (projected_imgs - tgt_imgs) * valid_points.float()
    reconstruction_loss = diff.abs().mean()

    # MS-SSIM loss
    ms_ssim_loss = 0.5 * (1 - ms_ssim(projected_imgs * valid_points.float(), tgt_imgs * valid_points.float(), data_range=255, size_average=True))
    '''
    # total loss
    loss = self.ld * abs_loss + self.lp * reconstruction_loss + self.ls * ms_ssim_loss
    return loss, t_loss, q_loss, reconstruction_loss, ms_ssim_loss

class MapNetCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False):
    """
    Implements L_D from eq. 2 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    """
    super(MapNetCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

  def forward(self, pred, targ):
    """
    :param pred: N x T x 6
    :param targ: N x T x 6
    :return:
    """

    # absolute pose loss
    s = pred.size()
    abs_t_loss = self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], 
                            targ.view(-1, *s[2:])[:, :3])
    abs_q_loss = self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                            targ.view(-1, *s[2:])[:, 3:])
                                            
    abs_loss =\
      torch.exp(-self.sax) * abs_t_loss + \
      self.sax + \
      torch.exp(-self.saq) * abs_q_loss + \
      self.saq

    # get the VOs
    pred_vos = pose_utils.calc_vos_simple(pred)
    targ_vos = pose_utils.calc_vos_simple(targ)

    # VO loss
    s = pred_vos.size()
    vo_t_loss = self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                              targ_vos.view(-1, *s[2:])[:, :3])
    vo_q_loss = self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                              targ_vos.view(-1, *s[2:])[:, 3:])
    vo_loss = \
      torch.exp(-self.srx) * vo_t_loss + \
      self.srx + \
      torch.exp(-self.srq) * vo_q_loss  + \
      self.srq

    # total loss
    loss = abs_loss + vo_loss
    return loss, abs_t_loss, abs_q_loss, vo_t_loss, vo_q_loss

class MapNetOnlineCriterion(nn.Module):
  def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0,
               saq=0.0, srx=0, srq=0.0, learn_beta=False, learn_gamma=False,
               gps_mode=False):
    """
    Implements L_D + L_T from eq. 4 in the paper
    :param t_loss_fn: loss function to be used for translation
    :param q_loss_fn: loss function to be used for rotation
    :param sax: absolute translation loss weight
    :param saq: absolute rotation loss weight
    :param srx: relative translation loss weight
    :param srq: relative rotation loss weight
    :param learn_beta: learn sax and saq?
    :param learn_gamma: learn srx and srq?
    :param gps_mode: If True, uses simple VO and only calculates VO error in
    position
    """
    super(MapNetOnlineCriterion, self).__init__()
    self.t_loss_fn = t_loss_fn
    self.q_loss_fn = q_loss_fn
    self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
    self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
    self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
    self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)
    self.gps_mode = gps_mode

  def forward(self, pred, targ):
    """
    targ contains N groups of pose targets, making the mini-batch.
    In each group, the first T poses are absolute poses, used for L_D while
    the next T-1 are relative poses, used for L_T
    All the 2T predictions in pred are absolute pose predictions from MapNet,
    but the last T predictions are converted to T-1 relative predictions using
    pose_utils.calc_vos()
    :param pred: N x 2T x 7
    :param targ: N x 2T-1 x 7
    :return:
    """
    s = pred.size()
    T = s[1] / 2
    pred_abs = pred[:, :T, :].contiguous()
    pred_vos = pred[:, T:, :].contiguous()  # these contain abs pose predictions for now
    targ_abs = targ[:, :T, :].contiguous()
    targ_vos = targ[:, T:, :].contiguous()  # contain absolute translations if gps_mode

    # absolute pose loss
    pred_abs = pred_abs.view(-1, *s[2:])
    targ_abs = targ_abs.view(-1, *s[2:])
    abs_loss =\
      torch.exp(-self.sax) * self.t_loss_fn(pred_abs[:, :3], targ_abs[:, :3]) + \
      self.sax + \
      torch.exp(-self.saq) * self.q_loss_fn(pred_abs[:, 3:], targ_abs[:, 3:]) + \
      self.saq

    # get the VOs
    if not self.gps_mode:
      pred_vos = pose_utils.calc_vos(pred_vos)

    # VO loss
    s = pred_vos.size()
    pred_vos = pred_vos.view(-1, *s[2:])
    targ_vos = targ_vos.view(-1, *s[2:])
    idx = 2 if self.gps_mode else 3
    vo_loss = \
      torch.exp(-self.srx) * self.t_loss_fn(pred_vos[:, :idx], targ_vos[:, :idx]) + \
      self.srx
    if not self.gps_mode:
      vo_loss += \
        torch.exp(-self.srq) * self.q_loss_fn(pred_vos[:, 3:], targ_vos[:, 3:]) + \
        self.srq

    # total loss
    loss = abs_loss + vo_loss
    return loss

def main():
  """
  print every loss term
  """

  from torch.utils import data
  from torchvision import transforms, models
  from optimizer import Optimizer
  from torch.autograd import Variable

  import sys
  sys.path.insert(0, '../')
  from dataset_loaders.composite import MF
  from models.posenet import PoseNet, MapNet

  dataset = '7Scenes'
  data_path = '../data/deepslam_data/7Scenes'
  seq = 'chess'
  steps = 3
  skip = 5
  # mode = 2: rgb and depth; 1: only depth; 0: only rgb
  mode = 2
  num_workers = 5
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),

  ])
  depth_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
	transforms.Lambda(lambda x: x.float())
  ])
  target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
  kwargs = dict(scene=seq, data_path=data_path, transform=transform,
                steps=steps, skip=skip)

  dset = MF(dataset=dataset, train=True, target_transform=target_transform,
            depth_transform=depth_transform, mode=mode, **kwargs)
  print 'Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
    len(dset))
  
  data_loader = data.DataLoader(dset, batch_size=5, shuffle=True,
    num_workers=num_workers)

  lr = 1e-4
  opt_method = 'adam'
  weight_decay = 0.0005
  dropout = 0.5
  criterion = GeoPoseNetCriterion()
  criterion.cuda()
  feature_extractor = models.resnet50(pretrained=True)
  posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True)
  model = MapNet(mapnet=posenet)
  model.train()
  model.cuda()
  param_list = [{'params': model.parameters()}]
  optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay)

  batch_count = 0
  N = 2
  for (data, target) in data_loader:
    # data: {'c': B x steps x 3 x H x W, 'd': B x steps x 1 x H x W}
    # target: B x steps x 6 translation + log q
    print 'Minibatch {:d}'.format(batch_count)
    color_var = Variable(data['c'], requires_grad=False).cuda(async=True)
    depth_var = Variable(data['d'], requires_grad=False).cuda(async=True)

    with torch.set_grad_enabled(True):
      output = model(color_var)

    target_var = Variable(target, requires_grad=False).cuda(async=True)
    with torch.set_grad_enabled(True):
      loss, abs_loss, reconstruction_loss, ms_ssim_loss = criterion(output, target_var, color_var, depth_var)

    optimizer.learner.zero_grad()
    loss.backward()
    optimizer.learner.step()

    print loss.item(), abs_loss.item(), reconstruction_loss.item(), ms_ssim_loss.item()

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
    main()