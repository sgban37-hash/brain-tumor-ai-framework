
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.optimizers import Novograd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
)
from monai.utils import set_determinism


# ============================================================
# PHASE 4 TRAINING
#
# Adapted from the official MONAI BraTS training workflow.
#
# MONAI source:
# Project-MONAI/tutorials
# acceleration/distributed_training/brats_training_ddp.py
#
# Apache-2.0 licensed.
#
# Project-specific components retained here:
# - our dataset_adapter
# - our 4-channel representation
# - our WT/TC/ET target representation
# - our SegResNet contract
# - our 96^3 patch configuration
# - our checkpoint structure
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))
import dataset_adapter as adapter


class MSDDataset(Dataset):

    def __init__(self, case_ids, data_root, transform=None):
        self.case_ids = list(case_ids)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):

        case_id = self.case_ids[index]

        # Our existing adapter is reused.
        # Override its dataset root so the framework works
        # independently of the original Windows path.
        adapter.MSD_ROOT = str(self.data_root)

        sample = adapter.load_msd(case_id)

        item = {
            "image": sample["image"],
            "target": sample["target"],
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


def build_transforms(train, patch_size):

    transforms = [
        EnsureTyped(
            keys=["image", "target"],
            dtype=torch.float32,
        )
    ]

    if train:

        transforms.extend([
            RandSpatialCropd(
                keys=["image", "target"],
                roi_size=patch_size,
                random_size=False,
            ),

            RandFlipd(
                keys=["image", "target"],
                prob=0.5,
                spatial_axis=0,
            ),

            RandFlipd(
                keys=["image", "target"],
                prob=0.5,
                spatial_axis=1,
            ),

            RandFlipd(
                keys=["image", "target"],
                prob=0.5,
                spatial_axis=2,
            ),
        ])

    transforms.append(
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True,
        )
    )

    if train:

        transforms.extend([
            RandScaleIntensityd(
                keys="image",
                factors=0.1,
                prob=0.5,
            ),

            RandShiftIntensityd(
                keys="image",
                offsets=0.1,
                prob=0.5,
            ),
        ])

    return Compose(transforms)


def evaluate(
    model,
    loader,
    device,
    patch_size,
    overlap,
):

    model.eval()

    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
    )

    post = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = batch["target"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                enabled=True,
            ):

                prediction = sliding_window_inference(
                    inputs=image,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )

            prediction = post(prediction)

            dice_metric(
                y_pred=prediction,
                y=target,
            )

    scores = (
        dice_metric
        .aggregate()
        .detach()
        .cpu()
        .numpy()
    )

    dice_metric.reset()

    return scores


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device("cuda")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed = 42

    set_determinism(
        seed=seed
    )

    torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    patch_size = (
        96,
        96,
        96,
    )

    data_root = Path(
        args.data_dir
    )

    image_dir = (
        data_root
        / "imagesTr"
    )

    # --------------------------------------------------------
    # Load the canonical deterministic MSD split manifest
    manifest_path = Path(__file__).resolve().parents[1] / "outputs" / "msd_split_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MSD split manifest not found: {manifest_path}"
        )

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    required_columns = {"case_id", "split"}

    if not split_rows or not required_columns.issubset(split_rows[0].keys()):
        raise RuntimeError(
            "MSD split manifest must contain 'case_id' and 'split' columns."
        )

    train_ids = [
        row["case_id"]
        for row in split_rows
        if row["split"] == "train"
    ]

    val_ids = [
        row["case_id"]
        for row in split_rows
        if row["split"] == "validation"
    ]

    test_ids = [
        row["case_id"]
        for row in split_rows
        if row["split"] == "internal_test"
    ]

    if len(train_ids) != 387 or len(val_ids) != 48 or len(test_ids) != 49:
        raise RuntimeError(
            f"Unexpected split sizes: "
            f"train={len(train_ids)}, "
            f"validation={len(val_ids)}, "
            f"internal_test={len(test_ids)}"
        )

    print(f"Training cases: {len(train_ids)}")
    print(f"Validation cases: {len(val_ids)}")
    print(f"Internal test cases reserved: {len(test_ids)}")

   # Dataset
    # --------------------------------------------------------

    train_dataset = MSDDataset(
        train_ids,
        data_root,
        transform=build_transforms(
            True,
            patch_size,
        ),
    )

    val_dataset = MSDDataset(
        val_ids,
        data_root,
        transform=build_transforms(
            False,
            patch_size,
        ),
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # --------------------------------------------------------
    # MODEL
    #
    # This preserves OUR model contract.
    # --------------------------------------------------------

    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=(
            1,
            2,
            2,
            4,
        ),
        blocks_up=(
            1,
            1,
            1,
        ),
        dropout_prob=0.2,
    ).to(device)

    # --------------------------------------------------------
    # LOSS
    #
    # Transplanted from MONAI's BraTS workflow.
    # --------------------------------------------------------

    loss_function = DiceFocalLoss(
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
        batch=True,
    )

    # --------------------------------------------------------
    # OPTIMIZER + SCHEDULER
    # --------------------------------------------------------

    optimizer = Novograd(
        model.parameters(),
        lr=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    # --------------------------------------------------------
    # CHECKPOINT DIRECTORY
    # --------------------------------------------------------

    checkpoint_dir = (
        REPO_ROOT
        / "outputs"
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_mean_dice = -1.0
    resume_path = checkpoint_dir / "last_model.pth"

    if resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        

        start_epoch = checkpoint["epoch"]
        best_mean_dice = checkpoint.get(
            "best_mean_dice",
            checkpoint.get("mean_dice", -1.0),
        )

        print(f"Resuming after epoch {start_epoch}")
    else:
        start_epoch = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            image = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                enabled=True,
            ):

                prediction = model(
                    image
                )

                loss = loss_function(
                    prediction,
                    target,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
            )

            print(
                f"Epoch "
                f"{epoch + 1}/"
                f"{args.epochs} | "
                f"Step "
                f"{step}/"
                f"{len(train_loader)} | "
                f"Loss "
                f"{loss.item():.4f}"
            )

        scheduler.step()

        average_loss = (
            running_loss
            / len(train_loader)
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"\nEpoch "
            f"{epoch + 1} complete | "
            f"Loss={average_loss:.4f} | "
            f"Time={elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "scaler_state_dict":
                scaler.state_dict(),
            "loss": average_loss,
            "best_mean_dice": best_mean_dice,
            },
            checkpoint_dir
            / "last_model.pth",
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            (epoch + 1)
            % args.val_interval
            == 0
        ):

            scores = evaluate(
                model,
                val_loader,
                device,
                patch_size,
                overlap=0.25,
            )

            mean_dice = float(
                np.mean(scores)
            )

            print(
                "\nValidation:"
            )

            print(
                f"Mean Dice: "
                f"{mean_dice:.4f}"
            )

            print(
                f"WT Dice: "
                f"{scores[0]:.4f}"
            )

            print(
                f"TC Dice: "
                f"{scores[1]:.4f}"
            )

            print(
                f"ET Dice: "
                f"{scores[2]:.4f}"
            )

            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------

            if (
                mean_dice
                > best_mean_dice
            ):

                best_mean_dice = (
                    mean_dice
                )

                torch.save(
                    {
                        "epoch":
                            epoch + 1,
                        "model_state_dict":
                            model.state_dict(),
                        "mean_dice":
                            best_mean_dice,
                        "channel_dice":
                            scores,
                    },
                    checkpoint_dir
                    / "best_model.pth",
                )

    print("\n")
    print("=" * 65)
    print(
        "PHASE 4 — "
        "10-CASE T4 SMOKE TEST COMPLETE"
    )
    print("=" * 65)

    print(
        f"Best validation mean Dice: "
        f"{best_mean_dice:.4f}"
    )


if __name__ == "__main__":
    main()
