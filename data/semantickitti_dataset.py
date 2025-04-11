import json
import os
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from data.base_dataset import get_lidar_rays, BaseDataset

LEARNING_MAP = {
                0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
                30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
                51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
                99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
            }
LEARNING_MAP_INV = {
                0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
                10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81
            }
COLOR_MAP = {
            0: [0, 0, 0], 1: [0, 0, 255], 10: [245, 150, 100], 11: [245, 230, 100], 13: [250, 80, 100], 15: [150, 60, 30], 16: [255, 0, 0], 18: [180, 30, 80], 20: [255, 0, 0],
            30: [30, 30, 255], 31: [200, 40, 255], 32: [90, 30, 150], 40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 49: [75, 0, 175], 50: [0, 200, 255], 51: [50, 120, 255],
            52: [0, 150, 255], 60: [170, 255, 150], 70: [0, 175, 0], 71: [0, 60, 135], 72: [80, 240, 150], 80: [150, 240, 255], 81: [0, 0, 255], 99: [255, 255, 50], 252: [245, 150, 100],
            256: [255, 0, 0], 253: [200, 40, 255], 254: [30, 30, 255], 255: [90, 30, 150], 257: [250, 80, 100], 258: [180, 30, 80], 259: [255, 0, 0]
        } # BGR
TRAINID2NAME = [
            "unlabeled", "car", "bicycle", "motorcycle", "truck", "other-veh.", "person", 
            "bicyclist", "m.cyclist", "road", "parking", "sidewalk", "other-gro.", "building", 
            "fence", "vegetation", "trunk", "terrain", "pole", "traffic-s."
        ]

@dataclass
class SemKITTIDataset(BaseDataset):
    device: str = "cpu"
    split: str = "train"  # train, val, test, (refine)
    root_path: str = "data/semantickitti"
    sequence_name: str = "00"
    sequence_id: str = "1050"
    lidar_type: str = "HDL-64E"
    preload: bool = True  # preload data into GPU
    scale: float = 1      # scale to bounding box
    offset: list = field(default_factory=list)  # offset
    fp16: bool = True     # if preload, load into fp16.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    num_rays_lidar: int = 4096
    fov_lidar: list = field(default_factory=list)
    num_frames: int = 50

    def __post_init__(self):
        frame_start = int(self.sequence_id)
        frame_end = frame_start + 49
        
        print(f"Using sequence {frame_start}-{frame_end}")
        self.frame_start = frame_start
        self.frame_end = frame_end

        self.training = self.split in ["train", "all", "trainval"]
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        if self.split == 'refine':
            self.split = 'train'
            self.num_rays_lidar = -1

        # load nerf-compatible format data.
        with open(
            os.path.join(self.root_path, 
                         f"transforms_{self.sequence_name}_{self.lidar_type}_{self.sequence_id}_{self.split}.json"),
            "r",
        ) as f:
            transform = json.load(f)

        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"])
            self.W = int(transform["w"])
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])

        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['lidar_file_path'])

        self.poses_lidar = []
        self.images_lidar = []
        self.images_depth = []
        self.times = []
        for f in tqdm.tqdm(frames, desc=f"Loading {self.split} data"):
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)

            f_lidar_path = os.path.join(self.root_path, f["lidar_file_path"])

            # channel1 None, channel2 intensity , channel3 depth , channel4 semantic
            pc = np.load(f_lidar_path)
            lidar_dim = pc.shape[-1]
            ray_drop = np.where(pc.reshape(-1, lidar_dim)[:, 2] == 0.0, 0.0, 1.0).reshape(
                self.H_lidar, self.W_lidar, 1
            )

            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale, pc[:, :, 3, None]],
                axis=-1,
            )
            image_depth = pc[:, :, 2][:, :, np.newaxis]

            time = np.asarray((f['frame_id']-frame_start)/(frame_end-frame_start))
            
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
            self.images_depth.append(image_depth)
            self.times.append(time)

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (
            self.poses_lidar[:, :3, -1] - self.offset
        ) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        self.images_lidar = torch.from_numpy(np.stack(self.images_lidar, axis=0)).float()  # [N, H, W, C]
        
        self.images_depth = torch.from_numpy(np.stack(self.images_depth, axis=0)).float()  # [N, H, W, 3]

        self.times = torch.from_numpy(np.asarray(self.times, dtype=np.float32)).view(-1, 1) # [N, 1]

        if self.preload:
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.fp16:
                dtype = torch.half
            else:
                dtype = torch.float
            self.images_lidar = self.images_lidar.to(dtype).to(self.device)
            self.images_depth = self.images_depth.to(dtype).to(self.device)
            self.times = self.times.to(self.device)

        self.intrinsics_lidar = self.fov_lidar

    def collate(self, index):
        B = len(index)  # a list of length 1

        results = {}

        poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]
        rays_lidar = get_lidar_rays(
            poses_lidar,
            self.intrinsics_lidar,
            self.H_lidar,
            self.W_lidar,
            self.num_rays_lidar,
            self.patch_size_lidar,
        )
        time_lidar = self.times[index].to(self.device) # [B, 1]

        images_lidar = self.images_lidar[index].to(self.device)  # [B, H, W, 4]
        images_depth = self.images_depth[index].to(self.device)  # [NV, H, W, 3]
        if self.training:
            C = images_lidar.shape[-1]
            images_lidar = torch.gather(
                images_lidar.view(B, -1, C),
                1,
                torch.stack(C * [rays_lidar["inds"]], -1),
            )  # [B, N, 3]

        results.update(
            {
                "H_lidar": self.H_lidar,
                "W_lidar": self.W_lidar,
                "rays_o_lidar": rays_lidar["rays_o"],
                "rays_d_lidar": rays_lidar["rays_d"],
                "rays_index": rays_lidar["inds"],
                "images_lidar": images_lidar,
                "images_depth": images_depth,
                "time": time_lidar,
                "poses_lidar": poses_lidar,
            }
        )

        return results

    def dataloader(self):
        size = len(self.poses_lidar)
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)
        return num_frames
