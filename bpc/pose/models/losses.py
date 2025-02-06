import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R  # For metric evaluation

#############################################
# Functions to Load Symmetry Data from JSON
#############################################

def generate_continuous_symmetries(axis, num_samples=12):
    """
    Generate discrete rotation matrices by sampling a continuous symmetry axis.
    
    Args:
        axis (list): A 3D unit vector representing the axis of rotation.
        num_samples (int): Number of discrete rotations to generate.
        
    Returns:
        sym_list (list of torch.Tensor): List of (3,3) rotation matrices.
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    angles = np.linspace(0, 2 * math.pi, num_samples, endpoint=False)
    sym_list = []
    for angle in angles:
        rot = R.from_rotvec(angle * axis)
        sym_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        sym_list.append(sym_matrix)
    return sym_list

def load_symmetry_from_json(json_path, num_samples=36):
    """
    Load both discrete and continuous symmetry transformations from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
        num_samples (int): Number of discrete samples for continuous symmetries.
        
    Returns:
        sym_dict (dict): Mapping from object ID (int) to list of (3,3) torch.Tensor rotations.
    """
    with open(json_path, "r") as f:
        json_data = json.load(f)

    sym_dict = {}
    for obj_id, obj_data in json_data.items():
        sym_list = []
        # Load discrete symmetries.
        for sym in obj_data.get("symmetries_discrete", []):
            sym_matrix = torch.tensor(sym).view(4, 4)[:3, :3]
            sym_list.append(sym_matrix)
        # Load continuous symmetries (if any).
        for sym in obj_data.get("symmetries_continuous", []):
            axis = sym["axis"]
            sym_list.extend(generate_continuous_symmetries(axis, num_samples))
        sym_dict[int(obj_id)] = sym_list
    return sym_dict

# Optionally, load a default symmetry file.
try:
    symmetry_json_path = os.path.join("datasets", "models", "models_info.json")
    symmetry_data = load_symmetry_from_json(symmetry_json_path, num_samples=12)
    print(f"[INFO] Loaded symmetry data from {symmetry_json_path}")
except Exception as e:
    print(f"[INFO] Could not load symmetry data from {symmetry_json_path}: {e}")
    symmetry_data = {}

#############################################
# Differentiable Helper Functions (PyTorch)
#############################################

def geodesic_distance_from_matrix(R1, R2):
    """
    Compute the geodesic distance (angle in radians) between two rotation matrices.
    R1, R2: (B, 3, 3)
    """
    R_rel = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    arg = (trace - 1) / 2
    arg = torch.clamp(arg, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(arg)
    return theta

def rotmat_from_euler(euler_angles):
    """
    Convert Euler angles (B,3) to rotation matrices (B,3,3) in a differentiable manner.
    Assumes intrinsic 'xyz' order and angles in radians.
    """
    cx = torch.cos(euler_angles[:, 0])
    sx = torch.sin(euler_angles[:, 0])
    cy = torch.cos(euler_angles[:, 1])
    sy = torch.sin(euler_angles[:, 1])
    cz = torch.cos(euler_angles[:, 2])
    sz = torch.sin(euler_angles[:, 2])
    
    R11 = cy * cz
    R12 = -cy * sz
    R13 = sy
    R21 = sx * sy * cz + cx * sz
    R22 = -sx * sy * sz + cx * cz
    R23 = -sx * cy
    R31 = -cx * sy * cz + sx * sz
    R32 = cx * sy * sz + sx * cz
    R33 = cx * cy
    
    R_mat = torch.stack([
        torch.stack([R11, R12, R13], dim=1),
        torch.stack([R21, R22, R23], dim=1),
        torch.stack([R31, R32, R33], dim=1)
    ], dim=1)
    return R_mat

def quat_to_rotmat(quat):
    """
    Convert quaternions (B,4) to rotation matrices (B,3,3).
    Assumes quaternion ordering [x, y, z, w].
    """
    quat = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = quat.unbind(dim=1)
    B = quat.shape[0]
    R_mat = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y,
    ], dim=1).view(B, 3, 3)
    return R_mat

def rotmat_from_6d(rep6d):
    """
    Convert 6D rotation representation to rotation matrices.
    Based on Zhou et al. (CVPR 2019).
    rep6d: (B,6) interpreted as two 3D vectors.
    """
    B = rep6d.shape[0]
    a1 = rep6d[:, 0:3]
    a2 = rep6d[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-8)
    tmp = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(tmp, dim=1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=1)
    R_mat = torch.stack([b1, b2, b3], dim=2)
    return R_mat

def euler_to_quat_torch(euler):
    """
    Convert a batch of Euler angles (B,3) to quaternions (B,4) using intrinsic 'xyz' order.
    Returns quaternions with ordering [x, y, z, w].
    """
    half = euler * 0.5
    cx = torch.cos(half[:, 0])
    sx = torch.sin(half[:, 0])
    cy = torch.cos(half[:, 1])
    sy = torch.sin(half[:, 1])
    cz = torch.cos(half[:, 2])
    sz = torch.sin(half[:, 2])
    
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    
    quat = torch.stack([x, y, z, w], dim=1)
    return quat

#############################################
# Helper Functions for Metric Evaluation (using SciPy)
#############################################

def compute_rot_deg_mean_quat(gt_quat, pred_quat):
    """
    Compute the mean angular error (in degrees) between batches of quaternions.
    Both gt_quat and pred_quat are torch tensors of shape (B, 4)
    (with ordering: [x, y, z, w]).
    """
    gt_quat_np = gt_quat.detach().cpu().numpy()
    pred_quat_np = pred_quat.detach().cpu().numpy()
    r1 = R.from_quat(gt_quat_np)
    r2 = R.from_quat(pred_quat_np)
    r_relative = r1.inv() * r2
    angles = r_relative.magnitude()
    return torch.tensor(np.mean(np.degrees(angles)), dtype=torch.float32)

def compute_rot_deg_mean_matrix(R_gt, R_pred):
    """
    Compute the mean angular error (in degrees) between batches of rotation matrices.
    Both R_gt and R_pred are torch tensors of shape (B, 3, 3).
    """
    R_gt_np = R_gt.detach().cpu().numpy()
    R_pred_np = R_pred.detach().cpu().numpy()
    r1 = R.from_matrix(R_gt_np)
    r2 = R.from_matrix(R_pred_np)
    r_relative = r1.inv() * r2
    angles = r_relative.magnitude()
    return torch.tensor(np.mean(np.degrees(angles)), dtype=torch.float32)

#############################################
# Loss Classes
#############################################

class EulerAnglePoseLoss(nn.Module):
    """
    Loss based on Euler angles.
    Assumes both predictions and groundtruth are in Euler angles (B, 3) in radians.
    Computes the geodesic distance between rotations.
    """
    def __init__(self, w_rot=1.0):
        super(EulerAnglePoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds, sym_list=None):
        pred_euler = preds[:, :3]
        gt_euler   = labels["euler"]
        pred_euler = torch.remainder(pred_euler + math.pi, 2 * math.pi) - math.pi
        gt_euler   = torch.remainder(gt_euler + math.pi, 2 * math.pi) - math.pi
        
        R_pred = rotmat_from_euler(pred_euler)
        R_gt   = rotmat_from_euler(gt_euler)
        
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        
        rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics

class QuaternionPoseLoss(nn.Module):
    """
    Loss using quaternion representation.
    The network prediction (first 4 numbers) is interpreted as a quaternion.
    Groundtruth quaternion is computed from the Euler label.
    """
    def __init__(self, w_rot=1.0):
        super(QuaternionPoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds, sym_list=None):
        gt_euler = labels["euler"]
        gt_quat = euler_to_quat_torch(gt_euler)
        
        pred_quat = preds[:, :4]
        eps = 1e-8
        pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + eps)
        gt_quat   = gt_quat / (gt_quat.norm(dim=1, keepdim=True) + eps)
        
        dot = (pred_quat * gt_quat).sum(dim=1, keepdim=True)
        pred_quat = torch.where(dot < 0, -pred_quat, pred_quat)
        
        R_pred = quat_to_rotmat(pred_quat)
        R_gt   = quat_to_rotmat(gt_quat)
        
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        
        rot_deg_mean = compute_rot_deg_mean_quat(gt_quat, pred_quat)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics

class SixDPoseLoss(nn.Module):
    """
    Loss using 6D rotation representation.
    The network prediction (first 6 numbers) is interpreted as the 6D representation.
    """
    def __init__(self, w_rot=1.0):
        super(SixDPoseLoss, self).__init__()
        self.w_rot = w_rot

    def forward(self, labels, preds, sym_list=None):
        gt_euler = labels["euler"]
        R_gt = rotmat_from_euler(gt_euler)
        rep6d = preds[:, :6]
        R_pred = rotmat_from_6d(rep6d)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rotation_loss = angles.mean()
        loss_val = self.w_rot * rotation_loss
        
        rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
        metrics = {
            "rot_loss": loss_val,
            "rot_deg_mean": rot_deg_mean,
        }
        return loss_val, metrics

class SymmetryAwarePoseLoss(nn.Module):
    """
    Symmetry-aware pose loss supporting Euler, Quaternion, and 6D representations.
    When the target object has symmetries, the loss is computed over all symmetric
    variants of the groundtruth rotation and the minimum error per sample is chosen.
    """
    def __init__(self, loss_type="euler", w_rot=1.0):
        super(SymmetryAwarePoseLoss, self).__init__()
        self.loss_type = loss_type
        self.w_rot = w_rot

    def forward(self, labels, preds, obj_id, sym_flag=True):
        # Retrieve symmetry transformations for the object if enabled.
        sym_list = symmetry_data.get(obj_id, []) if sym_flag else []
        
        # Convert groundtruth and predicted poses to rotation matrices.
        if self.loss_type == "euler":
            gt_euler = labels["euler"]
            R_gt = rotmat_from_euler(gt_euler)
            R_pred = rotmat_from_euler(preds[:, :3])
        elif self.loss_type == "quat":
            gt_quat = euler_to_quat_torch(labels["euler"])
            pred_quat = preds[:, :4]
            eps = 1e-8
            pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + eps)
            gt_quat = gt_quat / (gt_quat.norm(dim=1, keepdim=True) + eps)
            dot = (pred_quat * gt_quat).sum(dim=1, keepdim=True)
            pred_quat = torch.where(dot < 0, -pred_quat, pred_quat)
            R_gt = quat_to_rotmat(gt_quat)
            R_pred = quat_to_rotmat(pred_quat)
        elif self.loss_type == "6d":
            gt_euler = labels["euler"]
            R_gt = rotmat_from_euler(gt_euler)
            rep6d = preds[:, :6]
            R_pred = rotmat_from_6d(rep6d)
        else:
            raise ValueError("Invalid loss_type")

        # Compute the loss over symmetric poses.
        if sym_list:
            loss_list = []
            for S in sym_list:
                # Apply symmetry transformation: S is (3,3); R_gt is (B,3,3)
                R_gt_sym = torch.matmul(S.to(R_gt.device), R_gt)
                loss_sym = geodesic_distance_from_matrix(R_pred, R_gt_sym)  # (B,)
                loss_list.append(loss_sym)
                
            loss_stack = torch.stack(loss_list, dim=0)  # (num_sym, B)
            min_loss, _ = torch.min(loss_stack, dim=0)    # (B,)
            loss_val = min_loss.mean()
            # Compute the average rotation error (in degrees) per sample.
            rot_deg_mean = (min_loss * 180.0 / math.pi).mean()
        else:
            loss_tensor = geodesic_distance_from_matrix(R_pred, R_gt)  # (B,)
            loss_val = loss_tensor.mean()
            if self.loss_type in ["euler", "6d"]:
                rot_deg_mean = compute_rot_deg_mean_matrix(R_gt, R_pred)
            elif self.loss_type == "quat":
                rot_deg_mean = compute_rot_deg_mean_quat(gt_quat, pred_quat)
            else:
                rot_deg_mean = (loss_tensor * 180.0 / math.pi).mean().item()

        return self.w_rot * loss_val, {"rot_loss": loss_val, "rot_deg_mean": rot_deg_mean}
