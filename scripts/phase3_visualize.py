import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from monai.transforms import NormalizeIntensity
from monai.networks.nets import SegResNet


# ============================================================
# PATHS
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

OUTPUT_DIR = os.path.join(
    BASE,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "phase3_sanity_visualization.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading BRATS_001...")

mri = nib.load(MRI_PATH).get_fdata().astype(np.float32)

label = nib.load(LABEL_PATH).get_fdata().astype(np.int16)

print("MRI shape:", mri.shape)
print("Label shape:", label.shape)


# ============================================================
# CHANNEL CONVERSION
# ============================================================

# (H, W, D, 4)
#       ↓
# (4, H, W, D)

image = np.transpose(
    mri,
    (3, 0, 1, 2)
)


# ============================================================
# NORMALIZATION
# ============================================================

normalize = NormalizeIntensity(
    nonzero=True,
    channel_wise=True
)

image = np.asarray(
    normalize(image)
)


# ============================================================
# CREATE TARGET
# ============================================================

# WT = edema + non-enhancing + enhancing
# TC = non-enhancing + enhancing
# ET = enhancing

wt = (label > 0).astype(np.float32)

tc = np.logical_or(
    label == 2,
    label == 3
).astype(np.float32)

et = (label == 3).astype(np.float32)

target = np.stack(
    [wt, tc, et],
    axis=0
)


# ============================================================
# CENTER CROP
# ============================================================

roi = (96, 96, 96)

_, h, w, d = image.shape

start_h = (h - roi[0]) // 2
start_w = (w - roi[1]) // 2
start_d = (d - roi[2]) // 2

image_crop = image[
    :,
    start_h:start_h + roi[0],
    start_w:start_w + roi[1],
    start_d:start_d + roi[2]
]

target_crop = target[
    :,
    start_h:start_h + roi[0],
    start_w:start_w + roi[1],
    start_d:start_d + roi[2]
]


# ============================================================
# PYTORCH TENSOR
# ============================================================

image_tensor = torch.tensor(
    image_crop,
    dtype=torch.float32
).unsqueeze(0).to(device)


# ============================================================
# SEGRESNET
# ============================================================

print("\nCreating SegResNet...")

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


# ============================================================
# INFERENCE
# ============================================================

print("Running inference...")

with torch.no_grad():

    prediction = model(
        image_tensor
    )

prediction = torch.sigmoid(
    prediction
)

prediction = (
    prediction > 0.5
).float()

prediction = prediction.cpu().numpy()[0]


# ============================================================
# SELECT MIDDLE AXIAL SLICE
# ============================================================

slice_index = roi[2] // 2

flair_slice = image_crop[
    0,
    :,
    :,
    slice_index
]

ground_truth_slice = target_crop[
    0,
    :,
    :,
    slice_index
]

prediction_slice = prediction[
    0,
    :,
    :,
    slice_index
]


# ============================================================
# CREATE VISUALIZATION
# ============================================================

print("\nCreating visualization...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    "FLAIR — Input"
)

axes[0].axis("off")


axes[1].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].imshow(
    ground_truth_slice.T,
    alpha=0.45,
    origin="lower"
)

axes[1].set_title(
    "Ground Truth"
)

axes[1].axis("off")


axes[2].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    prediction_slice.T,
    alpha=0.45,
    origin="lower"
)

axes[2].set_title(
    "SegResNet Prediction"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3.2 VISUALIZATION COMPLETE")
print("=" * 60)

print("Saved to:")
print(OUTPUT_PATH)

print("=" * 60)

print("\nNOTE:")
print("The prediction is from an UNTRAINED model.")
print("It is only being used to verify the pipeline visually.")