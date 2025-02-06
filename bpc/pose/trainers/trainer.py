# bpc/pose/trainers/trainer.py

import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def batch_labels(label_list, device):
    """
    Given a list of label dictionaries (one per sample),
    stack each field into a batched tensor and move it to the device.
    """
    batched = {}
    for key in label_list[0]:
        batched[key] = torch.stack([d[key] for d in label_list], dim=0).to(device)
    return batched

def train_pose_estimation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=10,
    out_dir="checkpoints",
    device=torch.device("cuda"),
    resume=False,
):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    best_val_rot = float("inf")
    start_epoch = 1
    best_model_path = os.path.join(out_dir, "best_model.pth")
    checkpoint_path = os.path.join(out_dir, "last_checkpoint.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_rot = ckpt["best_val_rot"]

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_deg_sum  = 0.0
        train_steps = 0

        tbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}", ncols=120)
        for imgs, lbls, metas in tbar:
            sym_list = None
            imgs = imgs.to(device)
            batched_labels = batch_labels(lbls, device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, metrics = criterion(batched_labels, preds, sym_list=sym_list)
            loss.backward()
            optimizer.step()

            train_steps += 1
            train_loss_sum += metrics["rot_loss"].item()
            train_deg_sum  += metrics["rot_deg_mean"].item()

            tbar.set_postfix({
                "rot_loss": f"{metrics['rot_loss'].item():.3f}",
                "deg":      f"{metrics['rot_deg_mean'].item():.2f}",
            })

        train_loss_avg = train_loss_sum / train_steps
        train_deg_avg  = train_deg_sum  / train_steps

        writer.add_scalar("train/rot_loss", train_loss_avg, epoch)
        writer.add_scalar("train/rot_deg_mean", train_deg_avg, epoch)

        model.eval()
        val_loss_sum = 0.0
        val_deg_sum  = 0.0
        val_steps = 0

        with torch.no_grad():
            for imgs, lbls, metas in val_loader:
                sym_list = None
                imgs = imgs.to(device)
                batched_labels = batch_labels(lbls, device)
                loss, metrics = criterion(batched_labels, model(imgs), sym_list=sym_list)
                val_steps += 1
                val_loss_sum += metrics["rot_loss"].item()
                val_deg_sum  += metrics["rot_deg_mean"].item()

        val_loss_avg = val_loss_sum / val_steps
        val_deg_avg  = val_deg_sum  / val_steps

        writer.add_scalar("val/rot_loss", val_loss_avg, epoch)
        writer.add_scalar("val/rot_deg_mean", val_deg_avg, epoch)
        scheduler.step()
        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"TRAIN loss={train_loss_avg:.3f}, deg={train_deg_avg:.1f} | "
            f"VAL loss={val_loss_avg:.3f}, deg={val_deg_avg:.1f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rot": best_val_rot,
        }
        torch.save(ckpt, checkpoint_path)

        if val_deg_avg < best_val_rot:
            best_val_rot = val_deg_avg
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Saved best model with val_deg={best_val_rot:.2f}")

    writer.close()
    print("DONE. Best model =>", best_model_path)
    return model
