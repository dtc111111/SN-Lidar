#! /bin/bash
CUDA_VISIBLE_DEVICES=0
DATASET="semantickitti"
SEQ_NAME="00" 
SEQ_ID="1050"
LIDAR_TYPE="HDL-64E"

python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_name $SEQ_NAME --sequence_id $SEQ_ID --lidar_type $LIDAR_TYPE

python -m data.preprocess.lidar_to_nerf --dataset $DATASET --sequence_name $SEQ_NAME --sequence_id $SEQ_ID --lidar_type $LIDAR_TYPE

python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_name $SEQ_NAME --sequence_id $SEQ_ID --lidar_type $LIDAR_TYPE