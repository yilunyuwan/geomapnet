import torch
import torch.nn.functional as F
from pose_utils import qexp_t

'''
pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def logq2mat(quat):
    """Convert log quaternion coefficients to rotation matrix.

    Args:
        quat: three coeff of log quaternion of rotation.  -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = qexp_t(quat)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: -- [B, 6]
            for rotation_mode='euler': 6DoF parameters in the order of tx, ty, tz, rx, ry, rz 
            for rotation_mode='logq': tx, ty, tz, qvx, qvy, qvz
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points
'''

# -----------------------------------------

def q2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: four coeff of quaternion of rotation.  -- size = [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def generate_uv1_coords(b, h, w):
    """
    Generate [u, v, 1] coordinates 
    Args:
        b: batch_size
        h: image height
        w: image width
    Returns:
        uv_coords -- [B, 3, H, W]
    """
    u_range, v_range = torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h)
    grid_v, grid_u = torch.meshgrid([v_range, u_range])
    grid_ones = torch.ones(h, w)
    uv_coords = torch.stack((grid_u, grid_v, grid_ones), dim=0) # [3, H, W]
    uv_coords = uv_coords.expand(b, *uv_coords.size()) # [B, 3, H, W]
    return uv_coords

def generate_norm_coords(b, h, w):
    """
    Generate [u, v] coordinates normalized in the range of [-1, 1] 
    Args:
        b: batch_size
        h: image height
        w: image width
    Returns:
        uv_coords -- [B, H, W, 2]
    """
    u_range, v_range = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
    grid_v, grid_u = torch.meshgrid([v_range, u_range])
    uv_coords = torch.stack((grid_u, grid_v), dim=2) # [H, W, 2]
    uv_coords = uv_coords.expand(b, *uv_coords.size()) # [B, H, W, 2]
    return uv_coords

def generate_warp_coords(tgt_depths, relative_poses, depth_scale=1000, intrinsics=None):
    """
    Generate projection coordinates [u', v'], then intensity of pixel(u,v) of target image can be bilinear interpolated by the intensity of pixel(u', v') of source image
    Args:
        src_imgs: -- [B, 3, H, W]
        tgt_depths: -- [B, 1, H, W]
        relative_poses: -- [B, 7]
        intrinsic: -- [B, 3, 3]
    Returns:
        warp_coords: -- [B, H, W, 2]
    """
    b, _, h, w = tgt_depths.size()
    # Use default intrinsics, if intrinsics = None
    if (intrinsics is None):
        intrinsics = torch.tensor([[585, 0, 320],
                                    [0, 585, 240],
                                    [0, 0, 1]]).float()
    downscale_u = 640.0 / w
    downscale_v = 480.0 / h
    intrinsics = torch.cat((intrinsics[0:1, :]/downscale_u, intrinsics[1:2, :]/downscale_v, intrinsics[2:, :]), dim=0)
    intrinsics_inverse = intrinsics.inverse()

    uv_coords = generate_uv1_coords(b, h, w) # [B, 3, H, W]
    cam_coords = torch.matmul(intrinsics_inverse, uv_coords.reshape(b, 3, -1)) # [B, 3, H*W]
    # torch.set_printoptions(precision=8)
    # print cam_coords
    # depth map is a 16-bit image, and unit is millimeter.(except for TUM dataset whose depth_scale is 5000)
    cam_coords = cam_coords.type_as(tgt_depths) * tgt_depths.reshape(b, 1, h*w).expand(b, 3, h*w) / depth_scale # [B, 3, H*W]
    # print tgt_depths.reshape(b, 1, h*w)
    # print(tgt_depths.size())
    # a = 1.0
    # for i in cam_coords.size():
    #     a *= i
    # n = torch.sum(cam_coords == 0).float()
    # print a, a-n, (a-n)*1.0/a*100
    # print cam_coords
    translation = relative_poses[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot_mat = q2mat(relative_poses[:, 3:]) # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    proj_coords = torch.matmul(rot_mat, cam_coords) + translation
    warp_coords = torch.matmul(intrinsics.type_as(tgt_depths), proj_coords) # [B, 3, H*W]
    src_x, src_y, src_z = warp_coords[:, 0], warp_coords[:, 1], warp_coords[:, 2].clamp(min=1e-10) # [B, H*W]

    src_u = 2 * (src_x / src_z) / (w - 1) - 1
    src_v = 2 * (src_y / src_z) / (h - 1) - 1

    warp_coords = torch.stack([src_u, src_v], dim=2)  # [B, H*W, 2]
    return warp_coords.reshape(b, h, w, 2)

def calc_valid_points(depths, coords_norm, max_dist, max_depth=65535):
    b, _, h, w = depths.size()

    valid_depth_points = (depths < max_depth-10) * (depths > 10)
    valid_coords_points = coords_norm.abs().max(dim=-1)[0] <= 1
    valid_coords_points = valid_coords_points.unsqueeze(1)

    standard_coords = generate_norm_coords(b, h, w).type_as(depths) # [B, H, W, 2]
    valid_dist_points = (coords_norm - standard_coords).abs().max(dim=-1)[0] <= max_dist
    valid_dist_points = valid_dist_points.unsqueeze(1)
    valid_points = valid_depth_points * valid_coords_points * valid_dist_points

    return valid_points.float()

def reconstruction(imgs, depths, poses, depth_scale=1000, intrinsics=None, max_dist=0.8, max_depth=65535, padding_mode='zeros'):
    """
    Reconstruct the target image from a source image.

    Args:
        imgs: the source image (where to sample pixels) -- [B, 3, H, W]
        depths: depth map of the target image -- [B, 1, H, W]
        poses: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """

    b, _, h, w = imgs.size()

    # Get projection matrix for tgt camera frame to source pixel frame
    src_pixel_coords = generate_warp_coords(depths, poses, depth_scale, intrinsics)
    # print src_pixel_coords
    projected_imgs = F.grid_sample(imgs, src_pixel_coords, padding_mode=padding_mode)

    valid_points = calc_valid_points(depths, src_pixel_coords, max_dist, max_depth)
    

    # a = 1.0
    # for i in valid_points.size():
    #     a *= i
    # n1 = torch.sum(valid_depth_points == 0).float()
    # n2 = torch.sum(valid_coords_points == 0).float()
    # n3 = torch.sum(valid_points == 0).float()
    # print a, a-n1, a-n2, a-n3

    return projected_imgs, valid_points

def main():
  """
  visualizes the reconstruction
  """

  from torch.utils import data
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  from vis_utils import show_batch, show_stereo_batch, show_triplet_batch, show_triplet_depth_batch
  from pose_utils import calc_vo_logq2q
  import sys
  sys.path.insert(0, '../')
  from dataset_loaders.composite import MF
  from ssim import ssim, ms_ssim

  dataset = 'TUM'
  data_path = '../data/deepslam_data/TUM'
  seq = 'fr1'
  steps = 3
  skip = 1
  # mode = 2: rgb and depth; 1: only depth; 0: only rgb
  mode = 2
  num_workers = 6
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
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
  print 'Loaded TUM sequence {:s}, length = {:d}'.format(seq,
    len(dset))
  
  data_loader = data.DataLoader(dset, batch_size=5, shuffle=True,
    num_workers=num_workers)

  batch_count = 0
  N = 2
  for (imgs, poses) in data_loader:
    # imgs: {'c': B x steps x 3 x H x W, 'd': B x steps x 1 x H x W}
    # poses: B x steps x 6 translation + log q
    print 'Minibatch {:d}'.format(batch_count)
    color, depth = (imgs['c']+1)/2.0, (imgs['d']+1)/2.0
    # color, depth = imgs['c'], imgs['d']
    targ = poses
    s = poses.size()
    # get the photometric reconstruction
    # u_{src} = K T_{tgt->src} D_{tgt} K^{-1} u_{tgt}
    mid = s[1] / 2
    # src_imgs, tgt_imgs: (N*ceil(T/2)) x 3 x H x W 
    # tgt_depths: (N*ceil(T/2)) x 1 x H x W 
    src_imgs = color[:, :mid+1, ...].reshape(-1, *color.size()[2:])
    tgt_imgs = color[:, mid:, ...].reshape(-1, *color.size()[2:])
    src_depths = depth[:, :mid+1, ...].reshape(-1, *depth.size()[2:])
    tgt_depths = depth[:, mid:, ...].reshape(-1, *depth.size()[2:])
    # print src_imgs.size(), tgt_imgs.size(), tgt_depths.size()

    src_targ = targ[:, :mid+1, ...].reshape(-1, *s[2:])
    tgt_targ = targ[:, mid:, ...].reshape(-1, *s[2:])
    # print src_targ.size(), tgt_targ.size()
    # pred_relative_poses, targ_relative_poses: (N*ceil(T/2)) x 7
    targ_relative_poses = calc_vo_logq2q(src_targ, tgt_targ) 
    # print targ_relative_poses.shape
    # 7Scenes
    K = None
    depth_scale = 1000
    if dataset == 'TUM':
        depth_scale = 5000
        if seq == 'fr1':
            # TUM dataset
            K = torch.tensor([[517.3, 0, 318.6],
                                            [0, 516.5, 255.3],
                                            [0, 0, 1]]).float()
        else:
            # CoRBS dataset
            K = torch.tensor([[468.60, 0, 318.27],
                                            [0, 468.61, 243.99],
                                            [0, 0, 1]]).float()

    projected_imgs, valid_points = reconstruction(src_imgs, tgt_depths, targ_relative_poses, depth_scale=depth_scale, intrinsics=K)
    # print valid_points
    a = 1.0
    for i in valid_points.size():
        a *= i
    n = torch.sum(valid_points == True).float()         
    print a, n, n/a*100

    projected_imgs_valid = projected_imgs * valid_points
    tgt_imgs_valid = tgt_imgs * valid_points
    rgb_diff = torch.abs(projected_imgs_valid - tgt_imgs_valid)
    reconstruction_loss = torch.sum(rgb_diff) / torch.sum((valid_points>0).float()) / 3.0
    print 'reconstruction_loss under ground-truth pose:{:f}'.format(reconstruction_loss)

    imgx = projected_imgs * valid_points.float()
    imgy = tgt_imgs * valid_points.float()
    ssim_loss = 0.5 * (1 - ssim(imgx , imgy, data_range=1, win_size=3, size_average=True, mask=valid_points.float()))
    print 'ssim_loss under ground-truth pose:{:f}'.format(ssim_loss)

    # lb = make_grid(projected_imgs * valid_points.float(), nrow=mid+1, padding=25)
    lb = make_grid(src_imgs, nrow=mid+1, padding=25)
    # lb = make_grid(projected_imgs, nrow=mid+1, padding=25)
    mb = make_grid(tgt_imgs, nrow=mid+1, padding=25)
    rb = make_grid(projected_imgs*valid_points, nrow=mid+1, padding=25)

    # lb = make_grid(src_depths, normalize=True, scale_each=True, nrow=mid+1, padding=25)
    # mb = make_grid(tgt_depths, normalize=True, scale_each=True, nrow=mid+1, padding=25)
    # rb = make_grid(projected_imgs*valid_points.float(), normalize=True, scale_each=True, nrow=mid+1, padding=25)

    # show_triplet_depth_batch(lb, mb, rb)
    show_triplet_batch(lb, mb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
    main()