import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# PHASE 3.4 — FULL VOLUME EVALUATION + VISUALIZATION
# ============================================================

print("=" * 65)
print("PHASE 3.4 — FULL VOLUME EVALUATION")
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

LABEL_PATH = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour",
    "labelsTr",
    "BRATS_001.nii.gz"
)

PREDICTION_PATH = os.path.join(
    BASE,
    "outputs",
    "BRATS_001_phase3_full_prediction.nii.gz"
)

OUTPUT_PATH = os.path.join(
    BASE,
    "outputs",
    "BRATS_001_phase3_full_comparison.png"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\nChecking files...")

for path in [
    MRI_PATH,
    LABEL_PATH,
    PREDICTION_PATH
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

print("MRI         : FOUND")
print("Ground truth: FOUND")
print("Prediction  : FOUND")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading volumes...")

mri = nib.load(MRI_PATH).get_fdata()

ground_truth = nib.load(
    LABEL_PATH
).get_fdata().astype(np.uint8)

prediction = nib.load(
    PREDICTION_PATH
).get_fdata().astype(np.uint8)


print("MRI shape        :", mri.shape)
print("Ground truth     :", ground_truth.shape)
print("Prediction shape :", prediction.shape)


# ============================================================
# 4. CHECK SHAPES
# ============================================================

if ground_truth.shape != prediction.shape:

    raise ValueError(
        "Ground truth and prediction shapes do not match."
    )

print("Shape compatibility: PASS")


# ============================================================
# 5. CHECK LABELS
# ============================================================

print("\nGround truth labels:")
print(np.unique(ground_truth))

print("\nPrediction labels:")
print(np.unique(prediction))


# ============================================================
# 6. SELECT A REPRESENTATIVE SLICE
# ============================================================

# Find the slice containing the largest amount of
# ground-truth tumor.

tumor_per_slice = np.sum(
    ground_truth > 0,
    axis=(0, 1)
)

slice_index = int(
    np.argmax(tumor_per_slice)
)

print(
    "\nSelected axial slice:",
    slice_index
)

print(
    "Tumor pixels on slice:",
    tumor_per_slice[slice_index]
)


# ============================================================
# 7. EXTRACT SLICES
# ============================================================

# FLAIR is channel 0 in the MSD dataset.

flair_slice = mri[
    :,
    :,
    slice_index,
    0
]

ground_truth_slice = ground_truth[
    :,
    :,
    slice_index
]

prediction_slice = prediction[
    :,
    :,
    slice_index
]


# ============================================================
# 8. CREATE VISUALIZATION
# ============================================================

print("\nCreating comparison image...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# -------------------------
# MRI
# -------------------------

axes[0].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    "FLAIR — Input MRI"
)

axes[0].axis("off")


# -------------------------
# GROUND TRUTH
# -------------------------

axes[1].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].imshow(
    np.ma.masked_where(
        ground_truth_slice.T == 0,
        ground_truth_slice.T
    ),
    alpha=0.55,
    origin="lower"
)

axes[1].set_title(
    "Ground Truth"
)

axes[1].axis("off")


# -------------------------
# PREDICTION
# -------------------------

axes[2].imshow(
    flair_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    np.ma.masked_where(
        prediction_slice.T == 0,
        prediction_slice.T
    ),
    alpha=0.55,
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
# 9. SIMPLE WHOLE-TUMOR DICE
# ============================================================

gt_wt = ground_truth > 0
pred_wt = prediction > 0

intersection = np.logical_and(
    gt_wt,
    pred_wt
).sum()

gt_volume = gt_wt.sum()
pred_volume = pred_wt.sum()

if gt_volume + pred_volume == 0:

    dice = 1.0

else:

    dice = (
        2.0 * intersection
        / (gt_volume + pred_volume)
    )


print("\nWhole-tumor Dice:", dice)


# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 3.4 COMPLETE")
print("=" * 65)

print("Volume loading       : PASS")
print("Shape verification   : PASS")
print("Slice selection      : PASS")
print("Visualization        : PASS")
print("Dice calculation     : PASS")

print("=" * 65)

print("\nVisualization saved to:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print("The model is UNTRAINED.")
print("Therefore this Dice score is NOT model performance.")

print("=" * 65)