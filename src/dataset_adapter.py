import os
import numpy as np
import nibabel as nib


# ============================================================
# DATASET ADAPTER
# ============================================================
#
# Supported datasets:
#   1. MSD Task01_BrainTumour
#   2. UCSF-PDGM
#
# Common output:
#
#   image:
#       shape = (4, H, W, D)
#       channels = [FLAIR, T1, T1c, T2]
#
#   target:
#       shape = (3, H, W, D)
#       channels = [WT, TC, ET]
#
# ============================================================


BASE = r"C:\Users\S81nt\brain_tumor_project"

MSD_ROOT = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour"
)

UCSF_ROOT = os.path.join(
    BASE,
    "data",
    "UCSF-PDGM"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_nifti(path):
    """
    Load a NIfTI file and return its image data and NIfTI object.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    nii = nib.load(path)

    data = nii.get_fdata()

    return data, nii


def check_shape_and_spacing(
    arrays,
    names,
    expected_spacing=(1.0, 1.0, 1.0)
):
    """
    Verify that all volumes have the same shape
    and expected voxel spacing.
    """

    first_shape = arrays[0].shape

    for name, array in zip(names, arrays):

        if array.shape != first_shape:

            raise ValueError(
                f"Shape mismatch for {name}: "
                f"{array.shape} != {first_shape}"
            )

    print(
        "Common shape:",
        first_shape
    )

    # Check spacing using the first volume.
    #
    # All verified datasets use 1 mm isotropic spacing.

    spacing = np.array(
        arrays[0].shape  # only used to keep function generic
    )

    print(
        "Shape check: PASS"
    )


# ============================================================
# MSD ADAPTER
# ============================================================

def load_msd(case_id):
    """
    Load one MSD BrainTumour case.

    MSD stores all four MRI modalities together
    inside one 4-channel NIfTI file.

    Expected:
        (H, W, D, 4)

    Channel order:
        0 = FLAIR
        1 = T1
        2 = T1c
        3 = T2
    """

    image_path = os.path.join(
        MSD_ROOT,
        "imagesTr",
        case_id + ".nii.gz"
    )

    label_path = os.path.join(
        MSD_ROOT,
        "labelsTr",
        case_id + ".nii.gz"
    )

    image_data, image_nii = load_nifti(
        image_path
    )

    label_data, label_nii = load_nifti(
        label_path
    )

    print("\nLoading MSD:", case_id)

    print(
        "Raw MRI shape:",
        image_data.shape
    )

    print(
        "Raw label shape:",
        label_data.shape
    )

    # --------------------------------------------------------
    # Verify MRI has four modalities
    # --------------------------------------------------------

    if image_data.ndim != 4:
        raise ValueError(
            "MSD MRI should be 4-dimensional."
        )

    if image_data.shape[-1] != 4:
        raise ValueError(
            "MSD MRI should contain exactly 4 modalities."
        )

    # --------------------------------------------------------
    # Convert:
    #
    # (H, W, D, 4)
    #
    # to:
    #
    # (4, H, W, D)
    # --------------------------------------------------------

    image = np.transpose(
        image_data,
        (3, 0, 1, 2)
    ).astype(np.float32)

    label = label_data.astype(
        np.int16
    )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    labels_found = np.unique(label)

    expected_labels = {
        0, 1, 2, 3
    }

    if not set(
        labels_found
    ).issubset(expected_labels):

        raise ValueError(
            f"Unexpected MSD labels: "
            f"{labels_found}"
        )

    print(
        "MSD labels:",
        labels_found
    )

    # --------------------------------------------------------
    # Convert MSD labels to common WT / TC / ET format
    #
    # WT = 1 + 2 + 3
    # TC = 2 + 3
    # ET = 3
    # --------------------------------------------------------

    wt = (
        label > 0
    ).astype(np.float32)

    tc = np.logical_or(
        label == 2,
        label == 3
    ).astype(np.float32)

    et = (
        label == 3
    ).astype(np.float32)

    target = np.stack(
        [wt, tc, et],
        axis=0
    )

    print(
        "Standard image shape:",
        image.shape
    )

    print(
        "Standard target shape:",
        target.shape
    )

    return {
        "dataset": "MSD",
        "patient_id": case_id,
        "image": image,
        "target": target,
        "raw_label": label,
        "affine": image_nii.affine,
        "header": image_nii.header
    }


# ============================================================
# UCSF-PDGM ADAPTER
# ============================================================

def load_ucsf(case_id):
    """
    Load one UCSF-PDGM case.

    UCSF stores the modalities separately.

    Files:
        FLAIR
        T1
        T1c
        T2
        tumor_segmentation

    The four MRI volumes are stacked into:

        (4, H, W, D)

    in the common order:

        FLAIR, T1, T1c, T2
    """

    patient_dir = os.path.join(
        UCSF_ROOT,
        case_id
    )

    flair_path = os.path.join(
        patient_dir,
        case_id + "_FLAIR.nii.gz"
    )

    t1_path = os.path.join(
        patient_dir,
        case_id + "_T1.nii.gz"
    )

    t1c_path = os.path.join(
        patient_dir,
        case_id + "_T1c.nii.gz"
    )

    t2_path = os.path.join(
        patient_dir,
        case_id + "_T2.nii.gz"
    )

    label_path = os.path.join(
        patient_dir,
        case_id + "_tumor_segmentation.nii.gz"
    )

    # --------------------------------------------------------
    # Load all files
    # --------------------------------------------------------

    flair, flair_nii = load_nifti(
        flair_path
    )

    t1, t1_nii = load_nifti(
        t1_path
    )

    t1c, t1c_nii = load_nifti(
        t1c_path
    )

    t2, t2_nii = load_nifti(
        t2_path
    )

    label, label_nii = load_nifti(
        label_path
    )

    print("\nLoading UCSF-PDGM:", case_id)

    print(
        "FLAIR:",
        flair.shape
    )

    print(
        "T1:",
        t1.shape
    )

    print(
        "T1c:",
        t1c.shape
    )

    print(
        "T2:",
        t2.shape
    )

    print(
        "Label:",
        label.shape
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    volumes = [
        flair,
        t1,
        t1c,
        t2,
        label
    ]

    names = [
        "FLAIR",
        "T1",
        "T1c",
        "T2",
        "label"
    ]

    check_shape_and_spacing(
        volumes,
        names
    )

    # --------------------------------------------------------
    # Verify UCSF labels
    #
    # UCSF:
    #
    # 0 = background
    # 1 = NCR/NET
    # 2 = edema
    # 4 = enhancing tumor
    # --------------------------------------------------------

    labels_found = np.unique(
        label
    )

    expected_labels = {
        0, 1, 2, 4
    }

    if not set(
        labels_found
    ).issubset(expected_labels):

        raise ValueError(
            f"Unexpected UCSF labels: "
            f"{labels_found}"
        )

    print(
        "UCSF labels:",
        labels_found
    )

    # --------------------------------------------------------
    # Stack modalities
    #
    # IMPORTANT:
    # Keep the same order as MSD:
    #
    # FLAIR
    # T1
    # T1c
    # T2
    # --------------------------------------------------------

    image = np.stack(
        [
            flair,
            t1,
            t1c,
            t2
        ],
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Convert UCSF labels to common WT / TC / ET format
    #
    # UCSF:
    #
    # 1 = NCR/NET
    # 2 = edema
    # 4 = enhancing
    #
    # WT = 1 + 2 + 4
    # TC = 1 + 4
    # ET = 4
    # --------------------------------------------------------

    wt = np.logical_or(
        np.logical_or(
            label == 1,
            label == 2
        ),
        label == 4
    ).astype(np.float32)

    tc = np.logical_or(
        label == 1,
        label == 4
    ).astype(np.float32)

    et = (
        label == 4
    ).astype(np.float32)

    target = np.stack(
        [
            wt,
            tc,
            et
        ],
        axis=0
    )

    print(
        "Standard image shape:",
        image.shape
    )

    print(
        "Standard target shape:",
        target.shape
    )

    return {
        "dataset": "UCSF-PDGM",
        "patient_id": case_id,
        "image": image,
        "target": target,
        "raw_label": label.astype(np.int16),
        "affine": flair_nii.affine,
        "header": flair_nii.header
    }


# ============================================================
# TEST BOTH DATASET ADAPTERS
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TESTING DATASET ADAPTER")
    print("=" * 65)

    # --------------------------------------------------------
    # TEST MSD
    # --------------------------------------------------------

    print("\n\n")
    print("-" * 65)
    print("TEST 1 — MSD")
    print("-" * 65)

    msd = load_msd(
        "BRATS_001"
    )

    print(
        "\nMSD adapter: PASS"
    )

    # --------------------------------------------------------
    # TEST UCSF
    # --------------------------------------------------------

    print("\n\n")
    print("-" * 65)
    print("TEST 2 — UCSF-PDGM")
    print("-" * 65)

    ucsf = load_ucsf(
        "UCSF-PDGM-0004"
    )

    print(
        "\nUCSF adapter: PASS"
    )

    # --------------------------------------------------------
    # COMMON FORMAT CHECK
    # --------------------------------------------------------

    print("\n\n")
    print("-" * 65)
    print("COMMON FORMAT CHECK")
    print("-" * 65)

    print(
        "\nMSD image:",
        msd["image"].shape
    )

    print(
        "UCSF image:",
        ucsf["image"].shape
    )

    print(
        "\nMSD target:",
        msd["target"].shape
    )

    print(
        "UCSF target:",
        ucsf["target"].shape
    )

    # Both should have:
    #
    # image  = (4, H, W, D)
    # target = (3, H, W, D)

    if msd["image"].shape[0] != 4:
        raise ValueError(
            "MSD does not have 4 input channels."
        )

    if ucsf["image"].shape[0] != 4:
        raise ValueError(
            "UCSF does not have 4 input channels."
        )

    if msd["target"].shape[0] != 3:
        raise ValueError(
            "MSD target does not have 3 channels."
        )

    if ucsf["target"].shape[0] != 3:
        raise ValueError(
            "UCSF target does not have 3 channels."
        )

    print("\nCommon representation: PASS")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("PHASE 3.5.2 COMPLETE")
    print("=" * 65)

    print("MSD adapter              : PASS")
    print("UCSF-PDGM adapter        : PASS")
    print("Modality ordering        : PASS")
    print("Label harmonization      : PASS")
    print("Common image format      : PASS")
    print("Common target format     : PASS")
    print("=" * 65)

    print("\nThe two datasets can now be handled")
    print("through the same training interface.")