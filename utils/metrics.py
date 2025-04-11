import os
import numpy as np
import torch
import torch.nn.functional as F

import lpips
from skimage.metrics import structural_similarity
from sklearn.metrics import confusion_matrix

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.convert import pano_to_lidar
from data.semantickitti_dataset import TRAINID2NAME


def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean
    # distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


class DepthMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error, mask = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1
        
        return mask

    def compute_depth_errors(
        self, gt, pred, min_depth=1e-6, max_depth=80,
    ):  
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth
        
        rmse = (gt - pred) ** 2
        mask = rmse < 1
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

        return (rmse, medae, lpips_loss, ssim, psnr), mask

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error = {self.measure()}"


class IntensityMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
        self, gt, pred, min_intensity=1e-6, max_intensity=1.0,
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae =  np.median(np.abs(gt - pred))

        lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0), 
                                   torch.from_numpy(gt).squeeze(0), normalize=True).item()

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_intensity**2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Inten_error = {self.measure()}"


class RaydropMeter:
    def __init__(self, ratio=0.5):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        results = []

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        preds_mask = np.where(preds > self.ratio, 1, 0)
        acc = (preds_mask==truths).mean()
        results.append(acc)

        TP = np.sum((truths == 1) & (preds_mask == 1))
        FP = np.sum((truths == 0) & (preds_mask == 1))
        TN = np.sum((truths == 0) & (preds_mask == 0))
        FN = np.sum((truths == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append(f1)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Rdrop_error (RMSE, Acc, F1) = {self.measure()}"


class PointsMeter:
    def __init__(self, scale, intrinsics):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        pred_lidar = pano_to_lidar(preds[0], self.intrinsics)
        gt_lidar = pano_to_lidar(truths[0], self.intrinsics)

        dist1, dist2, idx1, idx2 = chamLoss(
            torch.FloatTensor(pred_lidar[None, ...]).cuda(),
            torch.FloatTensor(gt_lidar[None, ...]).cuda(),
        ) # len of dist1 may not equal to len of dist2
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"Point_error (CD, F-score) = {self.measure()}"
    

class SegmentationMeter:
    def __init__(self, num_classes):
        self.V = []
        self.N = 0
        self.num_classes = num_classes
        self.class_names = TRAINID2NAME

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

    def update(self, preds, truths, masks):
        preds, truths, masks = self.prepare_inputs(preds, truths, masks)
        
        B, H, W = preds.shape
        masks_np = np.array(masks)
        masks_np = masks_np.reshape(B, H, W)

        segmantation_error = self.compute_segmentation_errors(truths, preds, masks_np)
        self.V.append(segmantation_error)
        self.N += 1

    def compute_segmentation_errors(self, gt, pred, masks):
        
        gt_labels = np.unique(gt)
        gt = gt.flatten()
        pred = pred.flatten()
        masks = masks.flatten()
        
        cm = confusion_matrix(gt[masks], pred[masks], labels=np.arange(self.num_classes))

        # Pixel Accuracy
        pixel_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        # Mean IoU (mIoU)
        intersection = np.diag(cm)  # True Positive (TP)
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
        iou_per_class = np.zeros_like(intersection, dtype=float)
        iou_per_class[union > 0] = intersection[union > 0] / union[union > 0] # avoid division by zero
        iou_per_class = np.nan_to_num(iou_per_class, nan=0.0)
        mean_iou = np.nanmean(iou_per_class[gt_labels])

        return pixel_accuracy, mean_iou, iou_per_class

    def measure(self):
        assert self.N == len(self.V)
        
        PA = [v[0] for v in self.V]
        mIoU = [v[1] for v in self.V]
        iou_per_class = [v[2] for v in self.V]
        
        return np.mean(PA), np.mean(mIoU), np.mean(np.array(iou_per_class).reshape(len(self.V),-1),axis=0)

    def write(self, writer, global_step, prefix="", suffix=""):
        metrics = self.measure()
        writer.add_scalar(os.path.join(prefix, f"pixel_accuracy{suffix}"), metrics[0], global_step)
        writer.add_scalar(os.path.join(prefix, f"mean_iou{suffix}"), metrics[1], global_step)

    def report(self):
        metrics = self.measure()
        iou_per_class = metrics[2]
        
        report_str = f"Seman_error (PA, mIoU) = [{metrics[0]:.4f}, {metrics[1]:.4f}]\n"
        report_str += "== ↓ IoU per class ↓ ==\n"
        
        split_index = self.num_classes // 2
        class_names_part1 = self.class_names[:split_index]
        class_names_part2 = self.class_names[split_index:]
        iou_part1 = iou_per_class[:split_index]
        iou_part2 = iou_per_class[split_index:]
        
        report_str += " ".join([f"{class_name:<15}" for class_name in class_names_part1]) + "\n"
        report_str += " ".join([f"{iou:<15.4f}" for iou in iou_part1]) + "\n"
        report_str += " ".join([f"{class_name:<15}" for class_name in class_names_part2]) + "\n"
        report_str += " ".join([f"{iou:<15.4f}" for iou in iou_part2]) + "\n"
        return report_str
