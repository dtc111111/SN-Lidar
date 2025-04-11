import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def get_semantic_img(labels_trainid, dataset):
    """
    Get semantic images from labels
    :param labels_trainid: np.array of shape (H, W), train id
    :return semantic_img: np.array of shape (H, W, 3), BGR image
    """
    semantic_img = np.zeros((labels_trainid.shape[0], labels_trainid.shape[1], 3), dtype=np.uint8)
    if dataset == "semantickitti":
        from data.semantickitti_dataset import LEARNING_MAP_INV as learning_map_inv, COLOR_MAP as color_map
        inv_map_array = np.array([learning_map_inv[i] for i in range(20)])
        labels_id = inv_map_array[labels_trainid]
        for id, color in color_map.items():
            semantic_img[labels_id == id] = color
    semantic_img = semantic_img.astype(np.uint8)
    
    return semantic_img

def get_semantic_ply(points_with_labels_trainid, dataset):
    """
    Get semantic point cloud from labels
    :param labels_trainid: np.array of shape (H, W), train id
    :return semantic_ply: open3d.geometry.PointCloud, RGB color
    """
    semantic_ply = o3d.geometry.PointCloud()
    semantic_ply.points = o3d.utility.Vector3dVector(points_with_labels_trainid[:, :3])
    colors = np.zeros_like(points_with_labels_trainid[:, :3])
    if dataset == "semantickitti":
        from data.semantickitti_dataset import LEARNING_MAP_INV as learning_map_inv, COLOR_MAP as color_map
        inv_map_array = np.array([learning_map_inv[i] for i in range(20)])
        labels_id = inv_map_array[points_with_labels_trainid[:, 3].astype(np.int32)]
        for id, color in color_map.items():
            colors[labels_id == id] = color[2], color[1], color[0]
    semantic_ply.colors = o3d.utility.Vector3dVector(colors / 255)
    
    return semantic_ply

def get_ply(points):
    """
    Get semantic point cloud from labels
    :param labels_trainid: np.array of shape (H, W), train id
    :return semantic_ply: open3d.geometry.PointCloud, RGB color
    """
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points[:, :3])
    depth = points[:, 2]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=depth.min(), vmax=depth.max())
    colors = cmap(norm(depth))[:, :3]
    ply.colors = o3d.utility.Vector3dVector(colors)
    
    return ply
    