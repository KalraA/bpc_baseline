import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
from bpc.utils.data_utils import BOPSingleObjDataset, bop_collate_fn
from bpc.pose.models.simple_pose_net import SimplePoseNet
from bpc.pose.models.losses import (
    EulerAnglePoseLoss,
    QuaternionPoseLoss,
    SixDPoseLoss,
    SymmetryAwarePoseLoss,
    load_symmetry_from_json
)
from bpc.pose.trainers.trainer import train_pose_estimation
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to dataset root directory (with train_pbr and models_info.json)")
    parser.add_argument("--target_obj_id", type=int, default=11,
                        help="Target object ID")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint")
    parser.add_argument("--loss_type", type=str, default="euler", choices=["euler", "quat", "6d"],
                        help="Rotation loss type to use (and set model output dimension accordingly)")
    return parser.parse_args()

def find_scenes(root_dir):
    train_pbr_dir = os.path.join(root_dir, "train_pbr")
    if not os.path.exists(train_pbr_dir):
        raise FileNotFoundError(f"{train_pbr_dir} does not exist")
    all_items = os.listdir(train_pbr_dir)
    scene_ids = [item for item in all_items if item.isdigit()]
    scene_ids.sort()
    return scene_ids

def main():
    args = parse_args()

    scene_ids = find_scenes(args.root_dir)
    print(f"[INFO] Found scene_ids={scene_ids}")

    obj_id = args.target_obj_id
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"obj_{obj_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds_fixed = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=obj_id,
        target_size=256,
        augment=False,
        split="train"
    )
    train_ds_aug = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=obj_id,
        target_size=256,
        augment=True,
        split="train"
    )
    val_ds = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=obj_id,
        target_size=256,
        augment=False,
        split="val"
    )

    print(f"[INFO] train_ds_fixed:  {len(train_ds_fixed)} samples")
    print(f"[INFO] train_ds_aug:    {len(train_ds_aug)} samples")
    print(f"[INFO] val_ds:          {len(val_ds)} samples")

    train_dataset = ConcatDataset([train_ds_fixed, train_ds_aug])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pass loss_type so that the network outputs the correct dimension.
    model = SimplePoseNet(loss_type=args.loss_type, pretrained=not args.resume).to(device)

    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    step_size = max(1, args.epochs // 4)
    
    # Load symmetry data from JSON (data.json is assumed to be under the dataset root)
    symmetry_json_path = os.path.join(args.root_dir, "models", "models_info.json")
    symmetry_data = load_symmetry_from_json(symmetry_json_path, num_samples=12)
    
    # Determine if the current object has symmetry.
    if obj_id in symmetry_data and len(symmetry_data[obj_id]) > 0:
        print(f"[INFO] Object {obj_id} has symmetry. Switching to symmetry-aware loss.")
        criterion = SymmetryAwarePoseLoss(loss_type=args.loss_type)
        use_symmetry = True
    else:
        use_symmetry = False
        if args.loss_type == "euler":
            criterion = EulerAnglePoseLoss()
        elif args.loss_type == "quat":
            criterion = QuaternionPoseLoss()
        elif args.loss_type == "6d":
            criterion = SixDPoseLoss()
        else:
            raise ValueError("Invalid loss_type")
    
    # Wrap criterion if using symmetry-aware loss so that it receives obj_id and sym_flag.
    if use_symmetry:
        criterion_wrapper = lambda labels, preds, **kwargs: criterion(labels, preds, obj_id, sym_flag=True)
    else:
        criterion_wrapper = criterion

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_pose_estimation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion_wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        out_dir=checkpoint_dir,
        device=device,
        resume=args.resume
    )

if __name__ == "__main__":
    main()




"""
python3 train_pose.py \
  --root_dir datasets/ \
  --target_obj_id 8 \
  --epochs 50 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_workers 16 \
  --checkpoints_dir bpc/pose/pose_checkpoints/ \
  --loss_type quat

"""