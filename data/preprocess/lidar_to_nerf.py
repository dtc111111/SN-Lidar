import json
import argparse
import numpy as np
import os
from pathlib import Path
from .semantickitti_loader import SemKITTILoader

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="semantickitti",
        choices=["semantickitti", "kitti360"],
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
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    dataset = args.dataset
    sequence_name = args.sequence_name
    sequence_id = args.sequence_id
    lidar_type = args.lidar_type


    data_root = Path(__file__).parent.parent
    root_paths = {
        "kitti360": data_root / "kitti360" / "KITTI-360",
        "semantickitti": data_root / "semantickitti" / "dataset",
    }
    dataset_root = root_paths[dataset]
    dataset_parent_dir = dataset_root.parent

    s_frame_id = int(args.sequence_id)
    e_frame_id = s_frame_id + 49
    val_frame_ids = [s_frame_id + 9, s_frame_id + 19, s_frame_id + 29, s_frame_id + 39]

    print(f"Using sequence {sequence_name}: {s_frame_id}-{e_frame_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = val_frame_ids
    train_frame_ids = [x for x in frame_ids if x not in val_frame_ids]

    if dataset == "semantickitti":
        k3 = SemKITTILoader(dataset_root)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # Get lidar paths (range view not raw data).
    range_view_dir = dataset_parent_dir / "train" / sequence_name
    if dataset == "semantickitti":
        range_view_paths = [range_view_dir / "{:06d}.npy".format(int(frame_id)) for frame_id in frame_ids]

    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)

    # Get image dimensions, assume all images have the same dimensions.
    lidar_range_image = np.load(range_view_paths[0])
    lidar_h, lidar_w, _ = lidar_range_image.shape

    # Split by train/test/val.
    all_indices = [i - s_frame_id for i in frame_ids]
    train_indices = [i - s_frame_id for i in train_frame_ids]
    val_indices = [i - s_frame_id for i in val_frame_ids]
    test_indices = [i - s_frame_id for i in test_frame_ids]

    split_to_all_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        id_split = [frame_ids[i] for i in indices]
        lidar_paths_split = [range_view_paths[i] for i in indices]
        lidar2world_split = [lidar2world[i] for i in indices]

        json_dict = {
            "w_lidar": lidar_w,
            "h_lidar": lidar_h,
            "num_frames": num_frames,
            "num_frames_split": len(id_split),
            "frames": [
                {
                    "frame_id": id,
                    "lidar_file_path": str(
                        lidar_path.relative_to(dataset_parent_dir)
                    ),
                    "lidar2world": lidar2world.tolist(),
                }
                for (
                    id,
                    lidar_path,
                    lidar2world,
                ) in zip(
                    id_split,
                    lidar_paths_split,
                    lidar2world_split,
                )
            ],
        }
        json_path = dataset_parent_dir / f"transforms_{sequence_name}_{lidar_type}_{sequence_id}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
