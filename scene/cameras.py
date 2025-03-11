#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", K=None, 
                 sky_mask=None, normal=None, depth=None, camera_model=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 相机到世界的 C2W
        self.T = T  # 世界到相机的 W2C
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.sky_mask = sky_mask    # tensor: True为非天空区域，False为天空区域
        self.normal = normal    # None
        self.depth = depth  # tensor: H W

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)    # tensor, 归一化的 C H W
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        if camera_model == "SIMPLE_RADIAL" or "SIMPLE_PINHOLE":
            # fx = fy = K[0], cx = K[1], cy = K[2]
            self.K = torch.tensor([[K[0], 0, K[1]],
                                   [0, K[0], K[2]],
                                   [0, 0, 1]]).to(self.data_device).to(torch.float32)
        elif camera_model == "PINHOLE":
            # fx = K[0], fy = K[1], cx = K[2], cy = K[3]
            self.K = torch.tensor([[K[0], 0, K[2]],
                                   [0, K[1], K[3]],
                                   [0, 0, 1]]).to(self.data_device).to(torch.float32)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # W2C 世界到相机的变换矩阵
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()    # 生成了一个投影矩阵，用于将视图坐标投影到图像平面
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0) # 使用 bmm（批量矩阵乘法）将世界到视图变换矩阵和投影矩阵相乘，生成完整的投影变换矩阵
        self.camera_center = self.world_view_transform.inverse()[3, :3] # C2W的平移向量，即相机中心在世界坐标系中的位置

        if depth is not None:
            c2w = (self.world_view_transform.T).inverse()   # C2W，相机到世界坐标系的变换矩阵，cuda
            grid_x, grid_y = torch.meshgrid(torch.arange(self.image_width, device='cuda').float(), torch.arange(self.image_height, device='cuda').float(), indexing='xy')  # 生成1个二维网格，分别包含 x 轴和 y 轴的坐标
            points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)  # torch.stack 将 grid_x、grid_y 和一个全为 1 的张量按最后一个维度拼接，得到形状为 (H, W, 3) 的张量，每个位置的值是 (x, y, 1)
                                                                                                                   # reshape(-1, 3)将张量展平为形状为(H*W, 3)，即H*W个像素点在图像坐标系中的的齐次坐标，cuda
            # 每个像素点在世界坐标系下的坐标：像素坐标系 => 相机坐标系 => 世界坐标系。cuda
            rays_d = points @ self.K.inverse().T.cuda() @ c2w[:3, :3].T
            rays_o = c2w[:3, 3] # 相机中心在世界坐标系的位置（射线起点）。cuda

            # 计算每个3D点的世界坐标：深度值与射线方向相乘，并加上射线起点。cuda
            points = self.depth.cuda().reshape(-1, 1) * rays_d + rays_o
            points = points.reshape(*depth.shape[:], 3) # 重新调整为 H W 3

            output = torch.zeros_like(points)
            # 计算世界坐标系下3D点云在x、y方向上的梯度
            dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
            dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
            # 叉乘 得到法向量，然后归一化
            normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
            output[1:-1, 1:-1, :] = normal_map  # 中心填充
            self.normal = output.to(self.data_device).permute(2, 0, 1)  # 3 H W

        # # 显示深度图
        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        # import matplotlib as mpl
        # # 将图像从 PyTorch 张量转换为 NumPy 数组
        # depth_np = self.depth.detach().cpu().numpy()
        # image_np = self.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        # normal_np = self.normal.detach().cpu().numpy().transpose(1, 2, 0)
        # sky_mask_np = self.sky_mask.detach().cpu().numpy()
        # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # # 显示原始图像
        # axes[0].imshow(image_np)
        # axes[0].set_title('Original Image')
        #
        # # 显示深度图
        # depthmap = np.nan_to_num(depth_np)  # change nan to 0
        # constant_max = 999
        # constant_min = 0.1
        # normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
        # depth_vis_color = (mapper.to_rgba(depthmap)[:, :, :3] * 255).astype(np.uint8)
        # axes[1].imshow(depth_vis_color, cmap='gray')
        # axes[1].set_title('Depth Map')
        #
        # # 显示法线图
        # normal_np = (normal_np + 1.0) / 2.0
        # axes[2].imshow(normal_np)
        # axes[2].set_title('Normal Map')
        #
        # axes[3].imshow(sky_mask_np)
        # axes[3].set_title('Sky Mask')
        #
        # plt.show()
        # x = 0



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

