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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, normal_smooth_loss, depth_EdgeAwareLogL1, depth_smooth_loss
from utils.general_utils import vis_depth, read_propagted_depth
from gaussian_renderer import render, network_gui
from utils.graphics_utils import depth_propagation, check_geometric_consistency
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, load_pairs_relation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import numpy as np
import torchvision
from torchmetrics.functional.regression import pearson_corrcoef

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    #read the overlapping txt
    if opt.dataset == '360' and opt.depth_loss:
        pairs = load_pairs_relation(opt.pair_path)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # depth_loss_fn = ScaleAndShiftInvariantLoss(alpha=0.1, scales=1)
    propagated_iteration_begin = opt.propagated_iteration_begin
    propagated_iteration_after = opt.propagated_iteration_after
    after_propagated = False

    # 记录哪张图片未被更新过
    propagation_dict = {}
    for i in range(0, len(viewpoint_stack), 1):
        propagation_dict[viewpoint_stack[i].image_name] = False

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        randidx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack[randidx]
        
        # set the neighboring frames
        if opt.depth_loss:
            if opt.dataset == '360':
                src_idxs = pairs[randidx]
            else:
                if opt.dataset == 'waymo':
                    intervals = [-2, -1, 1, 2]
                elif opt.dataset == 'scannet':
                    intervals = [-10, -5, 5, 10]
                elif opt.dataset == 'free':
                    intervals = [-2, -1, 1, 2]
                src_idxs = [randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < len(viewpoint_stack)))]   # 当前训练相机的 候选邻居相机的索引

        # # 首先propagate高斯体_原版
        # with torch.no_grad():
        #     # 默认从1000至12000代，每20代传播一次（实际从1000至6000代，每50代一次）
        #     if opt.depth_loss and iteration > 1 and iteration < propagated_iteration_after:
        #     # if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):
        #         propagation_dict[viewpoint_cam.image_name] = True   # 记录该图片propagate过
        #
        #         # 返回从高斯体渲染的 depth 和 normal，不返回opacity
        #         render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
        #
        #         projected_depth = render_pkg["render_depth"]
        #
        #         # 在透明度<阈值的区域propagate depth。get the opacity that less than the threshold, propagate depth in these region
        #         # sky_mask 非天空区域为 True，天空区域为 False
        #         if viewpoint_cam.sky_mask is not None:
        #             sky_mask = viewpoint_cam.sky_mask.to(opacity_mask.device).to(torch.bool)
        #         else:
        #             sky_mask = None
        #
        #         # 进行传播操作
        #         depth_propagation(viewpoint_cam, projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
        #         # 读取传播后的 深度图（最大值为300）、得分图、法线图
        #         propagated_depth, cost, normal = read_propagted_depth('./cache/propagated_depth')
        #
        #         cost = torch.tensor(cost).to(projected_depth.device)
        #
        #         normal = torch.tensor(normal).to(projected_depth.device)
        #         # 使用W2C的旋转矩阵，将传播的 normal 转换到相机坐标系下，且恢复成 3 H W
        #         R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
        #         # R_w2c[:, 1:] *= -1
        #         normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)
        #
        #         propagated_depth = torch.tensor(propagated_depth).to(projected_depth.device)
        #         valid_mask = propagated_depth != 300    # 渲染深度值!= 300 则为True
        #
        #         # 计算渲染的深度图 和 传播的深度图的 绝对相对误差
        #         render_depth = render_pkg['render_depth']
        #
        #         abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
        #         # 绝对相对误差阈值：max(1.0)  - (max - min(0.8)) * (当前迭代次数 - propagate开始迭代次数) / (propagate结束迭代次数 - propagate开始迭代次数)
        #         abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - propagated_iteration_begin) / (propagated_iteration_after - propagated_iteration_begin)
        #
        #         # 计算渲染图像 和 原图像的 color误差
        #         render_color = render_pkg['render'].to(viewpoint_cam.data_device)
        #         color_error = torch.abs(render_color - viewpoint_cam.original_image)
        #         color_error = color_error.mean(dim=0).squeeze()
        #
        #         # for waymo, quantile 0.6; for free dataset, quantile 0.4
        #         # depth 误差 > 阈值，则为True
        #         error_mask = (abs_rel_error > abs_rel_error_threshold)
        #
        #         # 几何一致性筛选
        #         ref_K = viewpoint_cam.K     # 参考相机的内参
        #         ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()     # 参考相机的位姿
        #         geometric_counts = None
        #         # 遍历候选相机
        #         for idx, src_idx in enumerate(src_idxs):
        #             src_viewpoint = viewpoint_stack[src_idx]
        #             src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse() # 候选相机的位姿 C2W
        #             src_K = src_viewpoint.K
        #
        #             src_render_pkg = render(src_viewpoint, gaussians, pipe, bg, return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
        #             # 候选相机的 渲染深度图
        #             src_rendered_depth = src_render_pkg['render_depth']
        #
        #             # 候选相机的传播
        #             depth_propagation(src_viewpoint, torch.zeros_like(src_rendered_depth).cuda(), viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
        #             # 获取候选相机的传播深度图
        #             src_propagated_depth, cost, src_propagated_normal = read_propagted_depth('./cache/propagated_depth')
        #             src_propagated_depth = torch.tensor(src_propagated_depth).cuda()
        #             # 几何一致性检验（返回：有效mask，重投影点的在参考相机坐标系下的深度值，投影点在候选相机像素坐标系下的x，y坐标，重投影后的深度值与原深度值的相对误差）
        #             mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency( propagated_depth.unsqueeze(0), ref_K.unsqueeze(0).cuda(), ref_pose.unsqueeze(0),
        #                                                                                                           src_propagated_depth.unsqueeze(0), src_K.unsqueeze(0).cuda(), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
        #             if geometric_counts is None:
        #                 geometric_counts = mask.to(torch.uint8)
        #             else:
        #                 geometric_counts += mask.to(torch.uint8)
        #         # 累加与四个候选相机几何一致性检查后的得分，最大值为4，最小值为0
        #         cost = geometric_counts.squeeze()
        #         cost_mask = cost >= 2  # 通过两个及以上的候选相机几何一致性检查，则为True
        #
        #         # 将未通过几何一致性检查的像素点对应的 法向量置为 -10，即无效，3 H W
        #         normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
        #         viewpoint_cam.normal = normal  # 筛选后的法向量图 赋给gt_normal，viewpoint_cam.normal
        #
        #         # 传播mask：深度值有效 且 深度误差>阈值 且 几何一致性检查通过
        #         propagated_mask = valid_mask & error_mask & cost_mask
        #         if sky_mask is not None:
        #             propagated_mask = propagated_mask & sky_mask
        #
        #         # 传播mask的点超过100个，则将这些有深度的点 创建为初始高斯体
        #         if propagated_mask.sum() > 100:
        #             gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image)

        # 首先propagate高斯体_加载gt
        with torch.no_grad():
            # 默认从1000至12000代，每20代传播一次（实际从1000至6000代，每50代一次）
            if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):
                propagation_dict[viewpoint_cam.image_name] = True  # 记录该图片propagate过

                # 返回从高斯体渲染的 depth 和 normal，不返回opacity
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)

                rendered_depth = render_pkg["render_depth"]

                # 在透明度<阈值的区域propagate depth。get the opacity that less than the threshold, propagate depth in these region
                # sky_mask 非天空区域为 True，天空区域为 False
                if viewpoint_cam.sky_mask is not None:
                    sky_mask = viewpoint_cam.sky_mask.to(opacity_mask.device).to(torch.bool)
                else:
                    sky_mask = None

                propagated_normal = viewpoint_cam.normal.to(rendered_depth.device)
                propagated_depth = viewpoint_cam.depth.to(rendered_depth.device)

                propagated_depth[propagated_depth < 0] = 999
                propagated_depth[propagated_depth > 999] = 999
                valid_mask = propagated_depth < 999  # gt深度值 < 999，则为True

                # 渲染深度图 和 gt深度图的 绝对相对误差
                abs_rel_error = torch.abs(propagated_depth - rendered_depth) / propagated_depth
                # 绝对相对误差阈值（迭代次数越大 阈值越小）：max(1.0)  - (max - min(0.8)) * (当前迭代次数 - propagate开始迭代次数) / (propagate结束迭代次数 - propagate开始迭代次数)
                abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - propagated_iteration_begin) / (propagated_iteration_after - propagated_iteration_begin)
                # color error
                render_color = render_pkg['render'].to(viewpoint_cam.data_device)

                color_error = torch.abs(render_color - viewpoint_cam.original_image)
                color_error = color_error.mean(dim=0).squeeze()
                #for waymo, quantile 0.6; for free dataset, quantile 0.4
                # 深度误差 > 阈值，则为True
                error_mask = (abs_rel_error > abs_rel_error_threshold)

                # 几何一致性筛选
                ref_K = viewpoint_cam.K  # 参考相机的内参
                ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()  # 参考相机的位姿 C2W
                geometric_counts = None
                # 遍历候选相机
                for idx, src_idx in enumerate(src_idxs):
                    src_viewpoint = viewpoint_stack[src_idx]
                    src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse() # 候选相机的位姿 C2W
                    src_K = src_viewpoint.K

                    src_render_pkg = render(src_viewpoint, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                    src_projected_depth = src_render_pkg['render_depth']
                    
                    #get the src_depth first
                    depth_propagation(src_viewpoint, torch.zeros_like(src_projected_depth).cuda(), viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                    src_depth, cost, src_normal = read_propagted_depth('./cache/propagated_depth')
                    src_depth = torch.tensor(src_depth).cuda()
                    # 几何一致性检验（返回：有效mask，重投影点的在参考相机坐标系下的深度值，投影点在候选相机像素坐标系下的x，y坐标，重投影后的深度值与原深度值的相对误差）
                    mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0).cuda(),
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0),
                                                                                                                 src_K.unsqueeze(0).cuda(), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                    if geometric_counts is None:
                        geometric_counts = mask.to(torch.uint8)
                    else:
                        geometric_counts += mask.to(torch.uint8)
                # 累加与四个候选相机几何一致性检查后的得分，最大值为4，最小值为0
                cost = geometric_counts.squeeze()   # H W
                cost_mask = cost >= 2  # 通过两个及以上的候选相机几何一致性检查，则为True

                # 将未通过几何一致性检查的像素点对应的 法向量置为 -10，即无效，3 H W
                propagated_normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                viewpoint_cam.normal = propagated_normal  # 几何一致性筛选后的法向量图 作为gt_normal，viewpoint_cam.normal

                # 传播mask：深度值有效 且 深度误差>阈值 且 几何一致性检查通过
                propagated_mask = valid_mask & error_mask & cost_mask
                if sky_mask is not None:
                    propagated_mask = propagated_mask & sky_mask

                gt_image = viewpoint_cam.original_image.cuda()

                # 传播mask的点超过100个，则将这些有深度的点 创建为初始高斯体
                if propagated_mask.sum() > 100:
                    print("abs_rel_depth_error_threshold: {}, densify nums {}".format(abs_rel_error_threshold, propagated_mask.sum()))
                    gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # opacity mask
        # render_opacity：a-blending完成后 每个像素 对应的 对其渲染有贡献的 所有高斯累加的贡献度
        if iteration < opt.propagated_iteration_begin and opt.depth_loss:
            # < 1000代 且 计算loss
            opacity_mask = render_pkg['render_opacity'] > 0.999 # 很大，说明穿过的高斯少 或 不透明度低
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)    # (3,H,W)
        else:
            opacity_mask = render_pkg['render_opacity'] > 0.0   # 全部的mask
            opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=opacity_mask))
        # loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=opacity_mask))

        # flatten loss
        if opt.flatten_loss:
            scales = gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss += opt.lambda_flatten * flatten_loss
            # loss_flatten = 100.0 * flatten_loss

        if opt.normal_loss:
            rendered_normal = render_pkg['render_normal']
            if viewpoint_cam.normal is not None:
                normal_gt = viewpoint_cam.normal.cuda()

                if viewpoint_cam.sky_mask is not None:
                    filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10

                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()   # (H,W) --> (1,)
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean() # 1 - (H,W) --> (1,)
                loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal
                # loss_normal = 0.001 * l1_normal + 0.001 * cos_normal

        if iteration > opt.propagated_iteration_after:
        # if iteration >= 0:
            # FSGS
            rendered_depth = render_pkg["render_depth"]
            gt_depth = viewpoint_cam.depth.cuda()

            rendered_depth = rendered_depth.reshape(-1, 1)
            gt_depth = gt_depth.reshape(-1, 1)

            indx_d = gt_depth < 0.0   # <10%深度的 位置为True
            gt_depth[indx_d] = 0.0  # gt_depth中<10%深度的 值置为0
            rendered_depth[indx_d] = 0.0    # gt_depth中<10%深度的 值置为0

            ge_depth_mean = gt_depth[gt_depth<999].mean()
            depth_loss = min(
                (1 - pearson_corrcoef(- gt_depth, rendered_depth)),
                (1 - pearson_corrcoef(1 / (gt_depth + ge_depth_mean), rendered_depth))
            )
            loss += 0.02 * depth_loss

            rendered_depth = render_pkg["render_depth"]
            gt_depth = viewpoint_cam.depth.to(rendered_depth.device)
            # if viewpoint_cam.sky_mask is not None:
            #     filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
            # filter_mask_depth = torch.logical_and(gt_depth > 0.1, filter_mask)
            filter_mask_depth = gt_depth > 0.1
            l_depth = depth_EdgeAwareLogL1(rendered_depth, gt_depth.float(), gt_image, filter_mask_depth)
            l_depth_smooth = depth_smooth_loss(rendered_depth, filter_mask_depth)
            loss += 0.005 * (l_depth + 0.5 * l_depth_smooth)


        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # < 15000代，则进行增稠，Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # > 500 且 每100代增稠，则进行 增稠和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 初始为None，3000代后，即后续重置不透明度，则为20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # 每3000代 或 (白背景 且 为第500代)，则重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # < 30000代，优化，Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 第checkpoint_iterations 代时，保存相应代数的网络模型
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
