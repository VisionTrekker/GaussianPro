import os

GPU_ID = 1

# scenes = {"building1": "cuda", "building2": "cpu", "building3": "cpu", "town1": "cuda", "block2_sxfx": "cpu"}
scenes = {"block2_sxfx": "cpu"}
for idx, scene in enumerate(scenes.items()):
    ############ 训练 ############
    print("--------------------------------------------------------------")
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
            python train.py \
            -s ../../remote_data/dataset_simulator/{scene[0]}/train \
            -m output_gt_traingulate_1600/{scene[0]} \
            -r -1 \
            --data_device "{scene[1]}" \
            --port 6016 \
            --position_lr_init 0.00016 \
            --scaling_lr 0.005 \
            --percent_dense 0.01 \
            --flatten_loss \
            --normal_loss \
            --depth_loss \
            --propagation_interval 50 \
            --depth_error_min_threshold 0.8 \
            --depth_error_max_threshold 1.0 \
            --propagated_iteration_begin 1000 \
            --propagated_iteration_after 12000 \
            --patch_size 20 \
            --lambda_l1_normal 0.001 \
            --lambda_cos_normal 0.001 \
            --checkpoint_iterations 30000 \
            --test_iterations 2000 7000 15000 30000 \
            --save_iterations 15000 30000'
    print(cmd)
    os.system(cmd)

    ########### 渲染 ############
    # 可选：--skip_train --skip_test
    print("--------------------------------------------------------------")
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
            python render.py \
            -m output_gt_traingulate_1600/{scene[0]} \
            --skip_test'
    print(cmd)
    os.system(cmd)

    ############ 评测 ############
    print('---------------------------------------------------------------------------------')
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
                python metrics.py \
                -m output_gt_traingulate_1600/{scene[0]}'
    print(cmd)
    os.system(cmd)
