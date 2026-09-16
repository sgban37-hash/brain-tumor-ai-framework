import os
import glob
import numpy as np
import nibabel as nib


# ============================================================
# PHASE 3.5.3 — DATASET INTEGRITY VALIDATION
# ============================================================

print("=" * 70)
print("PHASE 3.5.3 — DATASET INTEGRITY VALIDATION")
print("=" * 70)


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


EXPECTED_SHAPE = (240, 240, 155)
EXPECTED_SPACING = (1.0, 1.0, 1.0)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_header(
    path,
    expected_shape,
    expected_spacing
):

    nii = nib.load(path)

    shape = nii.shape[:3]

    spacing = nii.header.get_zooms()[:3]

    orientation = nib.aff2axcodes(
        nii.affine
    )

    shape_ok = (
        shape == expected_shape
    )

    spacing_ok = np.allclose(
        spacing,
        expected_spacing,
        atol=1e-5
    )

    return (
        shape_ok,
        spacing_ok,
        spacing,
        orientation
    )


# ============================================================
# MSD VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING MSD")
print("-" * 70)

msd_images = sorted(
    glob.glob(
        os.path.join(
            MSD_ROOT,
            "imagesTr",
            "BRATS_*.nii.gz"
        )
    )
)

msd_labels = sorted(
    glob.glob(
        os.path.join(
            MSD_ROOT,
            "labelsTr",
            "BRATS_*.nii.gz"
        )
    )
)

print(
    "MRI files:",
    len(msd_images)
)

print(
    "Label files:",
    len(msd_labels)
)


msd_errors = []

for i, image_path in enumerate(msd_images):

    case_id = os.path.basename(
        image_path
    ).replace(
        ".nii.gz",
        ""
    )

    label_path = os.path.join(
        MSD_ROOT,
        "labelsTr",
        case_id + ".nii.gz"
    )

    if not os.path.exists(label_path):

        msd_errors.append(
            f"{case_id}: missing label"
        )

        continue

    # MRI header

    image_nii = nib.load(
        image_path
    )

    image_shape = image_nii.shape

    image_spacing = (
        image_nii.header.get_zooms()[:3]
    )

    image_orientation = nib.aff2axcodes(
        image_nii.affine
    )

    # Label header

    label_nii = nib.load(
        label_path
    )

    label_shape = label_nii.shape

    label_spacing = (
        label_nii.header.get_zooms()[:3]
    )

    label_orientation = nib.aff2axcodes(
        label_nii.affine
    )

    # Checks

    if image_shape[:3] != EXPECTED_SHAPE:

        msd_errors.append(
            f"{case_id}: MRI shape {image_shape}"
        )

    if label_shape != EXPECTED_SHAPE:

        msd_errors.append(
            f"{case_id}: label shape {label_shape}"
        )

    if not np.allclose(
        image_spacing,
        EXPECTED_SPACING,
        atol=1e-5
    ):

        msd_errors.append(
            f"{case_id}: MRI spacing {image_spacing}"
        )

    if not np.allclose(
        label_spacing,
        EXPECTED_SPACING,
        atol=1e-5
    ):

        msd_errors.append(
            f"{case_id}: label spacing {label_spacing}"
        )

    if image_orientation != label_orientation:

        msd_errors.append(
            f"{case_id}: orientation mismatch"
        )


print(
    "MSD validation errors:",
    len(msd_errors)
)

if len(msd_errors) == 0:

    print("MSD integrity: PASS")

else:

    print("MSD integrity: CHECK REQUIRED")

    for error in msd_errors[:20]:
        print("  ", error)


# ============================================================
# UCSF VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATING UCSF-PDGM")
print("-" * 70)


patient_dirs = sorted(
    [
        p
        for p in glob.glob(
            os.path.join(
                UCSF_ROOT,
                "UCSF-PDGM-*"
            )
        )
        if os.path.isdir(p)
    ]
)

print(
    "Patient directories:",
    len(patient_dirs)
)


ucsf_errors = []

for patient_dir in patient_dirs:

    patient_id = os.path.basename(
        patient_dir
    )

    files = {
        "FLAIR": os.path.join(
            patient_dir,
            patient_id + "_FLAIR.nii.gz"
        ),

        "T1": os.path.join(
            patient_dir,
            patient_id + "_T1.nii.gz"
        ),

        "T1c": os.path.join(
            patient_dir,
            patient_id + "_T1c.nii.gz"
        ),

        "T2": os.path.join(
            patient_dir,
            patient_id + "_T2.nii.gz"
        ),

        "label": os.path.join(
            patient_dir,
            patient_id + "_tumor_segmentation.nii.gz"
        )
    }

    headers = {}

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for name, path in files.items():

        if not os.path.exists(path):

            ucsf_errors.append(
                f"{patient_id}: missing {name}"
            )

            continue

        headers[name] = nib.load(path)

    if len(headers) != 5:
        continue

    # --------------------------------------------------------
    # Check shapes / spacing / orientation
    # --------------------------------------------------------

    reference_shape = None
    reference_spacing = None
    reference_orientation = None

    for name, nii in headers.items():

        shape = nii.shape[:3]

        spacing = (
            nii.header.get_zooms()[:3]
        )

        orientation = nib.aff2axcodes(
            nii.affine
        )

        if reference_shape is None:

            reference_shape = shape
            reference_spacing = spacing
            reference_orientation = orientation

        else:

            if shape != reference_shape:

                ucsf_errors.append(
                    f"{patient_id}: "
                    f"{name} shape mismatch"
                )

            if not np.allclose(
                spacing,
                reference_spacing,
                atol=1e-5
            ):

                ucsf_errors.append(
                    f"{patient_id}: "
                    f"{name} spacing mismatch"
                )

            if orientation != reference_orientation:

                ucsf_errors.append(
                    f"{patient_id}: "
                    f"{name} orientation mismatch"
                )

        if shape != EXPECTED_SHAPE:

            ucsf_errors.append(
                f"{patient_id}: "
                f"{name} shape {shape}"
            )

        if not np.allclose(
            spacing,
            EXPECTED_SPACING,
            atol=1e-5
        ):

            ucsf_errors.append(
                f"{patient_id}: "
                f"{name} spacing {spacing}"
            )


print(
    "UCSF validation errors:",
    len(ucsf_errors)
)

if len(ucsf_errors) == 0:

    print(
        "UCSF-PDGM integrity: PASS"
    )

else:

    print(
        "UCSF-PDGM integrity: CHECK REQUIRED"
    )

    for error in ucsf_errors[:20]:
        print("  ", error)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3.5.3 SUMMARY")
print("=" * 70)

print(
    "MSD cases checked:",
    len(msd_images)
)

print(
    "UCSF cases checked:",
    len(patient_dirs)
)

print(
    "MSD errors:",
    len(msd_errors)
)

print(
    "UCSF errors:",
    len(ucsf_errors)
)

print("=" * 70)

if len(msd_errors) == 0 and len(ucsf_errors) == 0:

    print(
        "\nDATASET INTEGRITY: PASS"
    )

else:

    print(
        "\nDATASET INTEGRITY: CHECK REQUIRED"
    )

print("=" * 70)