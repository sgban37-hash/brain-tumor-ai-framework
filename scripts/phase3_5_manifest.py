import os
import glob
import csv


# ============================================================
# PHASE 3.5.1 — DATASET MANIFEST
# ============================================================

print("=" * 65)
print("PHASE 3.5.1 — DATASET MANIFEST")
print("=" * 65)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE = r"C:\Users\S81nt\brain_tumor_project"

MSD_DIR = os.path.join(
    BASE,
    "data",
    "Task01_BrainTumour"
)

UCSF_DIR = os.path.join(
    BASE,
    "data",
    "UCSF-PDGM"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "dataset_manifest.csv"
)


# ============================================================
# 2. REQUIRED MODALITIES
# ============================================================

MODALITIES = [
    "FLAIR",
    "T1",
    "T1c",
    "T2"
]


# ============================================================
# 3. MSD CASE DISCOVERY
# ============================================================

print("\nScanning MSD dataset...")

msd_images_dir = os.path.join(
    MSD_DIR,
    "imagesTr"
)

msd_labels_dir = os.path.join(
    MSD_DIR,
    "labelsTr"
)

msd_cases = []

for path in sorted(
    glob.glob(
        os.path.join(
            msd_images_dir,
            "BRATS_*.nii.gz"
        )
    )
):

    filename = os.path.basename(path)

    case_id = filename.replace(
        ".nii.gz",
        ""
    )

    label_path = os.path.join(
        msd_labels_dir,
        case_id + ".nii.gz"
    )

    if os.path.exists(label_path):

        msd_cases.append({
            "dataset": "MSD",
            "patient_id": case_id,
            "FLAIR": path,
            "T1": path,
            "T1c": path,
            "T2": path,
            "label": label_path
        })


print(
    "MSD cases found:",
    len(msd_cases)
)


# ============================================================
# 4. UCSF CASE DISCOVERY
# ============================================================

print("\nScanning UCSF-PDGM dataset...")

ucsf_cases = []

patient_dirs = sorted(
    [
        p
        for p in glob.glob(
            os.path.join(
                UCSF_DIR,
                "UCSF-PDGM-*"
            )
        )
        if os.path.isdir(p)
    ]
)


for patient_dir in patient_dirs:

    patient_id = os.path.basename(
        patient_dir
    )

    flair = os.path.join(
        patient_dir,
        patient_id + "_FLAIR.nii.gz"
    )

    t1 = os.path.join(
        patient_dir,
        patient_id + "_T1.nii.gz"
    )

    t1c = os.path.join(
        patient_dir,
        patient_id + "_T1c.nii.gz"
    )

    t2 = os.path.join(
        patient_dir,
        patient_id + "_T2.nii.gz"
    )

    label = os.path.join(
        patient_dir,
        patient_id + "_tumor_segmentation.nii.gz"
    )

    required_files = [
        flair,
        t1,
        t1c,
        t2,
        label
    ]

    if all(
        os.path.exists(f)
        for f in required_files
    ):

        ucsf_cases.append({
            "dataset": "UCSF-PDGM",
            "patient_id": patient_id,
            "FLAIR": flair,
            "T1": t1,
            "T1c": t1c,
            "T2": t2,
            "label": label
        })


print(
    "UCSF-PDGM cases found:",
    len(ucsf_cases)
)


# ============================================================
# 5. COMBINE MANIFEST ENTRIES
# ============================================================

all_cases = (
    msd_cases +
    ucsf_cases
)


# ============================================================
# 6. SAVE MANIFEST
# ============================================================

print("\nWriting manifest...")

fieldnames = [
    "dataset",
    "patient_id",
    "FLAIR",
    "T1",
    "T1c",
    "T2",
    "label"
]

with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_cases
    )


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("MANIFEST SUMMARY")
print("=" * 65)

print(
    "MSD cases       :",
    len(msd_cases)
)

print(
    "UCSF-PDGM cases :",
    len(ucsf_cases)
)

print(
    "Total entries   :",
    len(all_cases)
)

print("=" * 65)

print("\nManifest saved to:")
print(MANIFEST_PATH)

print("\nIMPORTANT:")
print("The datasets were NOT copied.")
print("The datasets were NOT modified.")
print("This manifest only records where each sample is located.")

print("=" * 65)