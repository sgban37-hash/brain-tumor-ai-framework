import os
import numpy as np
import nibabel as nib
import torch

from monai.transforms import NormalizeIntensity
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric


# ============================================================
# PHASE 3 — END-TO-END AI SANITY RUN
# ============================================================

print("=" * 60)
print("PHASE 3 — END-TO-END AI SANITY RUN")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE = r"C:\Users\S81nt\brain_tumor_project"

MRI_PATH = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour",
    "imagesTr",
    "BRATS_001.nii.gz"
)

LABEL_PATH = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour",
    "labelsTr",
    "BRATS_001.nii.gz"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\nChecking files...")

if not os.path.exists(MRI_PATH):
    raise FileNotFoundError(
        f"MRI file not found:\n{MRI_PATH}"
    )

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(
        f"Label file not found:\n{LABEL_PATH}"
    )

print("MRI file  : FOUND")
print("Label file: FOUND")


# ============================================================
# 3. LOAD MRI AND GROUND TRUTH
# ============================================================

print("\nLoading BRATS_001...")

mri = nib.load(MRI_PATH).get_fdata().astype(np.float32)

label = nib.load(LABEL_PATH).get_fdata().astype(np.int16)

print("MRI shape      :", mri.shape)
print("Label shape    :", label.shape)
print("MRI datatype   :", mri.dtype)
print("Label datatype :", label.dtype)
print("Label values   :", np.unique(label))


# ============================================================
# 4. REARRANGE MRI CHANNELS
# ============================================================

# MSD stores:
#
# (H, W, D, 4)
#
# We need:
#
# (4, H, W, D)

print("\nRearranging MRI channels...")

image = np.transpose(
    mri,
    (3, 0, 1, 2)
)

print("Channel-first shape:", image.shape)


# ============================================================
# 5. MONAI INTENSITY NORMALIZATION
# ============================================================

print("\nApplying MONAI intensity normalization...")

normalize = NormalizeIntensity(
    nonzero=True,
    channel_wise=True
)

image = normalize(image)

print("Normalization: PASS")


# ============================================================
# 6. CREATE TARGET CHANNELS
# ============================================================

# MSD labels:
#
# 0 = Background
# 1 = Edema
# 2 = Non-enhancing tumor
# 3 = Enhancing tumor
#
# SegResNet output channels:
#
# Channel 0 = Whole Tumor (WT)
# Channel 1 = Tumor Core (TC)
# Channel 2 = Enhancing Tumor (ET)

print("\nCreating segmentation target channels...")

# Whole Tumor:
# edema + non-enhancing + enhancing

wt = (
    label > 0
).astype(np.float32)


# Tumor Core:
# non-enhancing + enhancing

tc = np.logical_or(
    label == 2,
    label == 3
).astype(np.float32)


# Enhancing Tumor

et = (
    label == 3
).astype(np.float32)


target = np.stack(
    [wt, tc, et],
    axis=0
)

print("Target shape:", target.shape)


# ============================================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================================

print("\nConverting to PyTorch tensors...")

# MONAI NormalizeIntensity may return a MetaTensor.
# np.asarray() safely converts it into a NumPy representation.

image_array = np.asarray(image)

target_array = np.asarray(target)

image_tensor = torch.tensor(
    image_array,
    dtype=torch.float32
).unsqueeze(0)

target_tensor = torch.tensor(
    target_array,
    dtype=torch.float32
).unsqueeze(0)

print("Input tensor :", image_tensor.shape)
print("Target tensor:", target_tensor.shape)


# ============================================================
# 8. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU   :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 9. CREATE 3D SEGRESNET
# ============================================================

print("\nCreating 3D SegResNet...")

model = SegResNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    init_filters=16,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2
).to(device)

model.eval()

print("SegResNet: CREATED")


# ============================================================
# 10. CREATE SMALL 3D CROP
# ============================================================

# RTX 3050 has 4 GB VRAM.
#
# Therefore we use a 96 × 96 × 96 crop.
#
# This is NOT training.
# It only checks whether the complete
# 3D AI pipeline can execute.

print("\nPreparing 96 × 96 × 96 sanity-test crop...")

roi_h = 96
roi_w = 96
roi_d = 96

_, _, h, w, d = image_tensor.shape

start_h = (h - roi_h) // 2
start_w = (w - roi_w) // 2
start_d = (d - roi_d) // 2

image_crop = image_tensor[
    :,
    :,
    start_h:start_h + roi_h,
    start_w:start_w + roi_w,
    start_d:start_d + roi_d
]

target_crop = target_tensor[
    :,
    :,
    start_h:start_h + roi_h,
    start_w:start_w + roi_w,
    start_d:start_d + roi_d
]

print("Image crop :", image_crop.shape)
print("Target crop:", target_crop.shape)


# ============================================================
# 11. MOVE DATA TO DEVICE
# ============================================================

print("\nMoving data to device...")

image_crop = image_crop.to(device)

target_crop = target_crop.to(device)

print("Data moved to:", device)


# ============================================================
# 12. RUN SEGRESNET
# ============================================================

print("\nRunning 3D SegResNet inference...")

with torch.no_grad():

    prediction = model(
        image_crop
    )

print("Inference: PASS")

print(
    "Raw prediction shape:",
    prediction.shape
)


# ============================================================
# 13. CONVERT OUTPUT TO PROBABILITIES
# ============================================================

print("\nApplying sigmoid...")

prediction_probability = torch.sigmoid(
    prediction
)

print(
    "Probability range:",
    prediction_probability.min().item(),
    "to",
    prediction_probability.max().item()
)


# ============================================================
# 14. CREATE BINARY SEGMENTATION
# ============================================================

print("\nCreating binary segmentation...")

prediction_binary = (
    prediction_probability > 0.5
).float()

print(
    "Binary prediction shape:",
    prediction_binary.shape
)


# ============================================================
# 15. CALCULATE DICE
# ============================================================

print("\nCalculating Dice...")

dice_metric = DiceMetric(
    include_background=True,
    reduction="mean"
)

dice_metric(
    y_pred=prediction_binary,
    y=target_crop
)

dice = dice_metric.aggregate().item()

dice_metric.reset()

print("Sanity Dice:", dice)


# ============================================================
# 16. CLEAR GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    del image_crop
    del target_crop
    del prediction
    del prediction_probability
    del prediction_binary

    torch.cuda.empty_cache()

    print("\nGPU memory cleared.")


# ============================================================
# 17. FINAL STATUS
# ============================================================

print("\n")
print("=" * 60)
print("PHASE 3 SANITY RUN COMPLETE")
print("=" * 60)

print("MRI loading         : PASS")
print("Channel conversion  : PASS")
print("MONAI normalization : PASS")
print("Target generation   : PASS")
print("PyTorch conversion  : PASS")
print("SegResNet creation  : PASS")
print("3D inference        : PASS")
print("Segmentation output : PASS")
print("Dice calculation    : PASS")

print("=" * 60)

print("\nIMPORTANT:")
print("The SegResNet is UNTRAINED.")
print("Therefore the Dice score is NOT model performance.")
print("This run only verifies technical pipeline execution.")

print("=" * 60)