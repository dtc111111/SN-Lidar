from pathlib import Path
import numpy as np


class SemKITTILoader:
    def __init__(self, semantic_kitti_root) -> None:
        # Root directory.
        self.semantic_kitti_root = Path(semantic_kitti_root)
        if not self.semantic_kitti_root.is_dir():
            raise FileNotFoundError(f"Semantic-KITTI {semantic_kitti_root} not found.")

    @staticmethod
    def parse_calibration(filename):
        """ read calibration file with given filename

                Returns
                -------
                dict
                        Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

                Returns
                -------
                list
                        list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def _load_all_lidars(self, sequence_name):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "00".

        Returns:
            velo_to_world: 4x4 metric.
        """
        data_poses_dir = self.semantic_kitti_root / "sequences" / sequence_name
        assert data_poses_dir.is_dir()
        
        calib_path = data_poses_dir / "calib.txt"
        calib = self.parse_calibration(calib_path)

        poses_path = data_poses_dir / "poses.txt"
        poses = self.parse_poses(poses_path, calibration=calib)
        
        velo_to_world_dict = dict()
        for i, pose in enumerate(poses):
            velo_to_world_dict[i] = pose

        return velo_to_world_dict

    def load_lidars(self, sequence_name, frame_ids):
        """
        Args:
            sequence_name: str, name of sequence. e.g. "00".
            frame_ids: list of int, frame ids. e.g. range(1050, 1099+1).

        Returns:
            velo_to_worlds
        """
        velo_to_world_dict = self._load_all_lidars(sequence_name)
        velo_to_worlds = []
        for frame_id in frame_ids:
            if frame_id in velo_to_world_dict.keys():
                velo_to_worlds.append(velo_to_world_dict[frame_id])
                tmp = velo_to_world_dict[frame_id]
            else:
                velo_to_worlds.append(tmp)
        velo_to_worlds = np.stack(velo_to_worlds)
        return velo_to_worlds
