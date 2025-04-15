<div align="center">

<h1>SN-LiDAR: Semantic Neural Fields for Novel Space-time View LiDAR Synthesis</h1>  

Yi Chen*, Tianchen Deng*, Wentao Zhao, Xiaoning Wang, Wenqian Xi, Weidong Chen, Jingchuan Wang† 

(* Equal contribution,† Corresponding author)  


**[Paper](https://arxiv.org/abs/2504.08361) | [Project Page]() | [Video]() | [Poster]()**

This repository is the official PyTorch implementation for SN-LiDAR.

<img src="./assets/main.png" alt="sn-lidar" width="70%">

</div>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
    </li>
    <li>
      <a href="#simulation">Simulation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>



## Demo

- Novel Space-time Synthesis on SemanticKITTI

<img src="./assets/novel.gif" width="80%"></img>

- Novel LiDAR Poses (X, Y, Z Shift)
<div>
  <img src="./assets/shift_x.gif" width="30%"/>
  <img src="./assets/shift_y.gif" width="30%"/>
  <img src="./assets/shift_z.gif" width="30%"/>
</div>

- Novel LiDAR Configuration (Beam change, Fov change)
<div>
  <img src="./assets/beams.gif" width="38%"/>
  <img src="./assets/fovs.gif" width="38.4%"/>
</div>

- Reconstruction of Dynamic Objects

<img src="./assets/dynamic.gif"></img>



## Overview
<img src="./assets/arch.png" width=100%>  

We propose SN-LiDAR, the first differential LiDAR-only framework for novel space-time LiDAR view synthesis with semantic labels, which achieves accurate semantic segmentation, high-quality geometric reconstruction, and realistic LiDAR synthesis. We integrate global geometric features from multi-resolution planar-grid representation with local semantic features from CNN-based semantic encoder. This fusion method not only strengthens the mutual enhancement between geometry and semantics but also enables processing large-scale scenes from coarse to fine.

<a id="getting-started"></a>

## Getting started


### 🛠️ Installation

```bash
git clone https://github.com/dtc111111/SN-Lidar.git
cd SN-Lidar

conda create -n sn-lidar python=3.9
conda activate sn-lidar

# PyTorch
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# compile packages in utils
cd utils/chamfer3D
python setup.py install
```


### 📁 Dataset

#### SemanticKITTI dataset ([Download](https://www.semantic-kitti.org/))
We use sequence 00-10 for experiments in our paper.   

Download SemanticKITTI dataset and put them into `data/semantickitti`.  
(or use symlinks: `ln -s DATA_ROOT/dataset ./data/semantickitti/`).  
The folder tree is as follows:  

```bash
data
└── semantickitti
    └── dataset
        └── sequences
            ├── 00
            │   ├── labels
            │   ├── velodyne
            │   ├── calib.txt
            │   ├── poses.txt
            │   └── times.txt
            └── 01
```

Next, run SemanticKITTI dataset preprocessing: 

```bash
# set $DATASET, $LIDAR_TYPE, $SEQ_NAME and $SEQ_ID
bash preprocess_data_semkitti.sh
```

After preprocessing, your folder structure should look like this:  

```bash
configs
├── semantickitti_{sequence_name}_{lidar_type}_{sequence_id}.txt
data
└── semantickitti
    ├── dataset
    ├── train
    ├── transforms_{sequence_name}_{lidar_type}_{sequence_id}_test.json
    ├── transforms_{sequence_name}_{lidar_type}_{sequence_id}_train.json
    └── transforms_{sequence_name}_{lidar_type}_{sequence_id}_val.json
```

### 🚀 Run SN-LiDAR

First, download pretrained [CENet](https://github.com/huixiancheng/CENet?tab=readme-ov-file#pretrained-models-and-logs) model trained on the KITTI dataset and place it with the path:
```
model/SalsaNext
```

Then, modify `run_kitti_sn.sh` by setting the appropriate values for `--config`, `--workspace`, and other parameters.
Then execute the script with:
```bash
bash run_kitti_sn.sh
```

<a id="simulation"></a>

## 🕹️ Simulation
After reconstruction, you can use the simulator to render and manipulate LiDAR point clouds in the whole scenario. It supports dynamic scene re-play, novel LiDAR configurations (`--fov_lidar`, `--H_lidar`, `--W_lidar`) and novel trajectory (`--shift_x`, `--shift_y`, `--shift_z`).    
Check the sequence config and corresponding workspace and model path (`--ckpt`).  
Run the following command:
```bash
bash run_kitti_sn_sim.sh
```
The results will be saved in the workspace folder.

## Acknowledgement
We would like to thank all the pioneers [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf), [LiDAR4D](https://github.com/ispc-lab/LiDAR4D), [CENet](https://github.com/huixiancheng/CENet).


## Citation
If your like our projects, please cite us and support us with a star 🌟.
<!-- ```bibtex

``` -->
