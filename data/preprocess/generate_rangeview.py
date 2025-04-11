import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import camtools as ct
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from collections import Counter


from utils.convert import lidar_to_pano_with_intensities


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="semantickitti",
        choices=["semantickitti", "kitti360", "nuScenes"],
        help="The dataset loader to use.",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        default="00",
        help="The sequence used to generate rangeview.",
    )
    parser.add_argument(
        "--sequence_id",
        type=str, 
        default="1050",
        help="choose start",
    )
    parser.add_argument(
        "--lidar_type",
        type=str,
        choices=["HDL-64E", "VLP-32C", "HDL-32E", "Livox-Mid360", "Livox-HAP"],
        help="The type of LiDAR sensor.",
    )
    parser.add_argument(
        "--use_semantic",
        action="store_true",
        help="Generate semantic rangeview.",
    )
    return parser

def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0, lidar_type="HDL-64E"
):
    pano, intensities, labels = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 4))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    range_view[:, :, 3] = labels
    return range_view


def generate_train_data(
    H,
    W,
    intrinsics,
    lidar_paths,
    out_dir,
    points_dim,
    max_depth=80.0,
    lidar_type="HDL-64E",
    label_paths=None,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if label_paths is not None:
        label_dict = {os.path.basename(label_path).split('.')[0].zfill(10): label_path for label_path in label_paths}

    for lidar_path in tqdm(lidar_paths):
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        intensity = point_cloud[:, 3]
        min_intensity = np.min(intensity)
        max_intensity = np.max(intensity)
        normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
        point_cloud[:, 3] = normalized_intensity
        
        lidar_file_name = os.path.basename(lidar_path).split(".")[0].zfill(10)
        if label_paths is not None:
            label_path = label_dict.get(lidar_file_name, None)
            if os.path.exists(label_path) and label_path:
                label = np.fromfile(label_path, dtype=np.uint32)
                label = label & 0xFFFF
            else:
                label = np.zeros((point_cloud.shape[0], 1), dtype=np.uint32)
        else:
            label = np.zeros((point_cloud.shape[0], 1), dtype=np.uint32)
        
        point_cloud_with_label = np.hstack([point_cloud, label.reshape(-1, 1).astype(np.uint32)])
        # pano[:, :, 0] no data，pano[:, :, 1] intensity，pano[:, :, 2] depth, pano[:, :, 3] label
        pano = LiDAR_2_Pano_KITTI(point_cloud_with_label, H, W, intrinsics, max_depth, lidar_type)
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(out_dir / frame_name, pano)


def create_kitti_rangeview(dataset, sequence_name, lidar_type, frame_start, frame_end):
    data_root = Path(__file__).parent.parent

    root_paths = {
        "kitti360": data_root / "kitti360" / "KITTI-360",
        "semantickitti": data_root / "semantickitti" / "dataset",
    }
    dataset_root = root_paths[dataset]
    dataset_parent_dir = dataset_root.parent
    out_dir = dataset_parent_dir / "train" / sequence_name

    if lidar_type == "HDL-64E":
        H = 66
        W = 1030
        intrinsics = (2.0, 26.9) # (fov_up, fov)
        max_depth = 80.0
    elif lidar_type == "HDL-32E":
        H = 32
        W = 1080
        intrinsics = (10.0, 40.0)
        max_depth = 100.0

    sequence_dir = sequence_name
    frame_ids = list(range(frame_start, frame_end + 1))
    lidar_dir = (
        dataset_root
        / "sequences"
        / sequence_dir
        / "velodyne"
    )
    lidar_paths = [os.path.join(lidar_dir, "%06d.bin" % frame_id) for frame_id in frame_ids]
    label_dir = (
        dataset_root
        / "sequences"
        / sequence_dir
        / "labels"
    )
    label_paths = [
        os.path.join(label_dir, "%06d.label" % frame_id) for frame_id in frame_ids
    ]

    generate_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        lidar_paths=lidar_paths,
        out_dir=out_dir,
        points_dim=4,
        max_depth=max_depth,
        lidar_type=lidar_type,
        label_paths=label_paths,
        frame_start=frame_start,
        frame_end=frame_end,
    )

def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    frame_start = int(args.sequence_id)
    frame_end = frame_start + 49     
            
    print(f"Generate {args.lidar_type} rangeview dataset: {args.dataset} seq: {args.sequence_name} from {frame_start} to {frame_end} ...")
    create_kitti_rangeview(args.dataset, args.sequence_name, args.lidar_type, frame_start, frame_end)


if __name__ == "__main__":
    main()
