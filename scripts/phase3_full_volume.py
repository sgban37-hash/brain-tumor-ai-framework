import os
import numpy as np
import nibabel as nib
import torch

from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity


# ============================================================
# PHASE 3.3 — FULL VOLUME SLIDING-WINDOW INFERENCE
# ============================================================

print("=" * 65)
print("PHASE 3.3 — FULL VOLUME SLIDING-WINDOW INFERENCE")
print("=" * 65)


# ============================================================
# 1. PATHS
# ============================================================

BASE = r"C:\Users\S81nt\brain_tumor_project"

MRI_PATH = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour",
    "imagesTr",
    "BRATS_001.nii.gz"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "BRATS_001_phase3_full_prediction.nii.gz"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. LOAD MRI
# ============================================================

print("\nLoading BRATS_001...")

nii = nib.load(MRI_PATH)

mri = nii.get_fdata().astype(
    np.float32
)

print(
    "Original MRI shape:",
    mri.shape
)


# ============================================================
# 4. CHANNEL-FIRST
# ============================================================

print("\nConverting channels...")

# (240, 240, 155, 4)
#       ↓
# (4, 240, 240, 155)

image = np.transpose(
    mri,
    (3, 0, 1, 2)
)

print(
    "Channel-first shape:",
    image.shape
)


# ============================================================
# 5. NORMALIZATION
# ============================================================

print("\nApplying MONAI normalization...")

normalize = NormalizeIntensity(
    nonzero=True,
    channel_wise=True
)

image = np.asarray(
    normalize(image)
)

print("Normalization: PASS")


# ============================================================
# 6. PYTORCH TENSOR
# ============================================================

image_tensor = torch.tensor(
    image,
    dtype=torch.float32
).unsqueeze(0)

print(
    "Input tensor:",
    image_tensor.shape
)


# ============================================================
# 7. CREATE SEGRESNET
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
# 8. MOVE INPUT TO GPU
# ============================================================

image_tensor = image_tensor.to(device)


# ============================================================
# 9. FULL-VOLUME SLIDING WINDOW
# ============================================================

print("\nStarting sliding-window inference...")

print("ROI size: 96 × 96 × 96")
print("Batch size: 1")
print("Overlap: 25%")

with torch.no_grad():

    prediction = sliding_window_inference(
        inputs=image_tensor,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
        mode="constant"
    )


print("\nSliding-window inference: PASS")

print(
    "Prediction tensor:",
    prediction.shape
)


# ============================================================
# 10. CONVERT TO PROBABILITIES
# ============================================================

prediction = torch.sigmoid(
    prediction
)


# ============================================================
# 11. BINARY SEGMENTATION
# ============================================================

prediction_binary = (
    prediction > 0.5
).float()


print(
    "Binary prediction:",
    prediction_binary.shape
)


# ============================================================
# 12. CONVERT TO SINGLE-LABEL MASK
# ============================================================

# The three output channels are:
#
# Channel 0 = Whole Tumor
# Channel 1 = Tumor Core
# Channel 2 = Enhancing Tumor
#
# For saving a simple segmentation mask,
# we use the most confident channel.
#
# 0 = background
# 1 = WT
# 2 = TC
# 3 = ET

prediction_cpu = prediction_binary.cpu().numpy()[0]

wt = prediction_cpu[0]
tc = prediction_cpu[1]
et = prediction_cpu[2]

single_mask = np.zeros(
    wt.shape,
    dtype=np.uint8
)

# Priority:
# Enhancing Tumor > Tumor Core > Whole Tumor

single_mask[wt > 0] = 1
single_mask[tc > 0] = 2
single_mask[et > 0] = 3


# ============================================================
# 13. SAVE NIFTI PREDICTION
# ============================================================

print("\nSaving prediction...")

prediction_nii = nib.Nifti1Image(
    single_mask,
    nii.affine,
    nii.header
)

nib.save(
    prediction_nii,
    OUTPUT_PATH
)

print(
    "Saved:",
    OUTPUT_PATH
)


# ============================================================
# 14. REPORT
# ============================================================

print("\nPrediction mask shape:", single_mask.shape)

print(
    "Prediction labels:",
    np.unique(single_mask)
)


# ============================================================
# 15. CLEAR GPU
# ============================================================

if torch.cuda.is_available():

    del image_tensor
    del prediction
    del prediction_binary

    torch.cuda.empty_cache()

    print("\nGPU memory cleared.")


# ============================================================
# 16. FINAL STATUS
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 3.3 COMPLETE")
print("=" * 65)

print("Full MRI loading          : PASS")
print("4-channel preprocessing   : PASS")
print("SegResNet                 : PASS")
print("Sliding-window inference : PASS")
print("Full-volume prediction    : PASS")
print("NIfTI output              : PASS")

print("=" * 65)

print("\nIMPORTANT:")
print("The model is UNTRAINED.")
print("This prediction is NOT a measure of model accuracy.")
print("This test verifies full-volume technical execution.")

print("=" * 65)