#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_snlidar.py \
--config configs/semantickitti_00_HDL-64E_1050.txt \
--workspace log/semKITTI/semKITTI_00 \
--name snlidar \
--lr 1e-2 \
--num_rays_lidar 1024 \
--iters 30000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
--alpha_s 0.01 \
--num_semantic_classes 20 \
--eval_interval 100 \