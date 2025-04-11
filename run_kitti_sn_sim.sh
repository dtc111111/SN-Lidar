#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_snlidar_sim.py \
--config configs/semantickitti_00_HDL-64E_1050.txt \
--workspace log/semKITTI/semKITTI_00/simulation \
--ckpt log/semKITTI/semKITTI_00/checkpoints/snlidar_ep1305_refine.pth \
--num_semantic_classes 20 \
--fov_lidar 2.0 26.9 \
--H_lidar 66 \
--W_lidar 1030 \
--shift_x 1.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--align_axis \
# --kitti2nus