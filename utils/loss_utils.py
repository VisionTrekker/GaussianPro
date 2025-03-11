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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask=mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = ssim_map[mask]

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor
    
def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

class MSELoss(torch.nn.Module):
    def __init__(self, reduction='image-based'):
        super().__init__()

        if reduction == 'image-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)
    
class GradientLoss(torch.nn.Module):
    def __init__(self, scales=4, reduction='image-based'):
        super().__init__()

        if reduction == 'image-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class ScaleAndShiftInvariantLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='image-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def LogL1(pred_depth, gt_depth):
    return torch.log(1 + torch.abs(pred_depth - gt_depth))


def depth_EdgeAwareLogL1(pred_depth, gt_depth, gt_image, valid_mask):
    logl1 = LogL1(pred_depth, gt_depth)

    gt_image = gt_image.permute(1, 2, 0)
    grad_img_x = torch.mean(torch.abs(gt_image[:-1, :, :] - gt_image[1:, :, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(gt_image[:, :-1, :] - gt_image[:, 1:, :]), -1, keepdim=True)
    lambda_x = torch.exp(-grad_img_x)
    lambda_y = torch.exp(-grad_img_y)

    x = logl1[:-1, :]
    loss_x = lambda_x * logl1[:-1, :].unsqueeze(-1)
    loss_y = lambda_y * logl1[:, :-1].unsqueeze(-1)

    if valid_mask is not None:
        assert valid_mask.shape == pred_depth.shape
        loss_x = loss_x[valid_mask[:-1, :]].mean()
        loss_y = loss_y[valid_mask[:, :-1]].mean()
    return loss_x + loss_y


def depth_smooth_loss(depth_map, filter_mask):
    # 计算深度图中相邻像素之间的差异
    diff_x = torch.abs(depth_map[:-1, :] - depth_map[1:, :])
    diff_y = torch.abs(depth_map[:, :-1] - depth_map[:, 1:])

    # 过滤掉异常值
    filter_mask_x = filter_mask[:-1, :] & filter_mask[1:, :]
    filter_mask_y = filter_mask[:, :-1] & filter_mask[:, 1:]

    # 计算总的平滑损失
    smooth_loss_x = diff_x[filter_mask_x].mean()
    smooth_loss_y = diff_y[filter_mask_y].mean()
    smooth_loss = smooth_loss_x + smooth_loss_y

    return smooth_loss

def normal_smooth_loss(normal_map,  filter_mask):
    # 将法线图从 (C, H, W) 变为 (H, W, C)
    normal_map = normal_map.permute(1, 2, 0)

    # 计算法线图中相邻像素之间的差异
    diff_x = torch.abs(normal_map[:-1, :, :] - normal_map[1:, :, :])
    diff_y = torch.abs(normal_map[:, :-1, :] - normal_map[:, 1:, :])

    # 过滤掉异常值
    filter_mask_x = filter_mask[:-1, :] & filter_mask[1:, :]
    filter_mask_y = filter_mask[:, :-1] & filter_mask[:, 1:]
    # 计算总的平滑损失
    smooth_loss_x = diff_x[filter_mask_x].mean()
    smooth_loss_y = diff_y[filter_mask_y].mean()
    smooth_loss = smooth_loss_x + smooth_loss_y

    return smooth_loss