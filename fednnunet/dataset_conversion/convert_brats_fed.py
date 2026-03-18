from __future__ import annotations

import argparse
import gzip
import json
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw


MODALITY_MAP = {
    "t1n": 0,
    "t1c": 1,
    "t2f": 2,
    "t2w": 3,
}


def remap_brats_segmentation(seg_src: Path, seg_dst: Path) -> None:
    """
    Remap BraTS labels from {0, 2, 3, 4} to {0, 1, 2, 3}.
    """
    img = nib.load(str(seg_src))
    data = img.get_fdata()

    remapped = np.zeros_like(data, dtype=np.uint8)
    remapped[data == 2] = 1
    remapped[data == 3] = 2
    remapped[data == 4] = 3

    remapped_img = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(remapped_img, str(seg_dst))
    
    
def sanitize_case_id(case_name: str) -> str:
    """Remove hyphens so the nnU-Net case identifier stays simple."""
    return case_name.replace("-", "")


def gzip_copy(src: Path, dst: Path) -> None:
    """Copy a .nii file to .nii.gz without modifying image content."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def create_dataset_json(dataset_dir: Path, dataset_name: str, num_training_cases: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "T1n",
            "1": "T1c",
            "2": "T2f",
            "3": "T2w",
        },
        "labels": {
            "background": 0,
            "NCR": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": dataset_name,
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


def get_case_dirs(input_dir: Path) -> List[Path]:
    case_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir() and p.name.startswith("BraTS-")]
    if not case_dirs:
        raise RuntimeError(f"No BraTS case directories found in {input_dir}")
    return case_dirs


def validate_case_dir(case_dir: Path) -> None:
    required = ["seg", "t1n", "t1c", "t2f", "t2w"]
    missing = []
    for suffix in required:
        expected = case_dir / f"{case_dir.name}-{suffix}.nii"
        if not expected.exists():
            missing.append(expected.name)
    if missing:
        raise FileNotFoundError(f"Missing required files in {case_dir.name}: {missing}")


def convert_case(case_dir: Path, images_tr: Path, labels_tr: Path) -> None:
    validate_case_dir(case_dir)
    case_id = sanitize_case_id(case_dir.name)

    # Copy modalities
    for modality, channel_idx in MODALITY_MAP.items():
        src = case_dir / f"{case_dir.name}-{modality}.nii"
        dst = images_tr / f"{case_id}_{channel_idx:04d}.nii.gz"
        gzip_copy(src, dst)

    # Copy segmentation
    seg_src = case_dir / f"{case_dir.name}-seg.nii"
    seg_dst = labels_tr / f"{case_id}.nii.gz"
    remap_brats_segmentation(seg_src, seg_dst)


def convert_split(case_dirs: List[Path], dataset_id: int, dataset_suffix: str) -> None:
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_suffix}"
    dataset_dir = Path(nnUNet_raw) / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(labels_tr)

    for case_dir in case_dirs:
        convert_case(case_dir, images_tr, labels_tr)

    create_dataset_json(dataset_dir, dataset_name, len(case_dirs))
    print(f"Created {dataset_name} with {len(case_dirs)} training cases at {dataset_dir}")


def split_cases_round_robin(case_dirs: List[Path], num_sites: int) -> List[List[Path]]:
    splits = [[] for _ in range(num_sites)]
    for i, case_dir in enumerate(case_dirs):
        splits[i % num_sites].append(case_dir)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw BraTS case folders",
    )
    parser.add_argument(
        "--start_dataset_id",
        type=int,
        default=301,
        help="Starting nnU-Net dataset ID",
    )
    parser.add_argument(
        "--num_sites",
        type=int,
        default=2,
        help="Number of simulated federated sites/datasets to create",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="BraTS",
        help="Dataset suffix prefix, e.g. BraTS -> Dataset301_BraTSA",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    case_dirs = get_case_dirs(input_dir)
    splits = split_cases_round_robin(case_dirs, args.num_sites)

    for idx, split in enumerate(splits):
        suffix = f"{args.dataset_prefix}{chr(ord('A') + idx)}"
        dataset_id = args.start_dataset_id + idx
        convert_split(split, dataset_id, suffix)


if __name__ == "__main__":
    main()
    
"""
rm -rf nnUNet_data/nnUNet_raw/Dataset301_BraTSA
rm -rf nnUNet_data/nnUNet_raw/Dataset302_BraTSB
rm -rf nnUNet_data/nnUNet_preprocessed/Dataset301_BraTSA
rm -rf nnUNet_data/nnUNet_preprocessed/Dataset302_BraTSB
rm -rf nnUNet_data/nnUNet_results/Dataset301_BraTSA
rm -rf nnUNet_data/nnUNet_results/Dataset302_BraTSB

python fednnunet/dataset_conversion/convert_brats_fed.py -i data --start_dataset_id 301 --num_sites 2

find nnUNet_data/nnUNet_raw/Dataset301_BraTSA/labelsTr -type f | wc -l
find nnUNet_data/nnUNet_raw/Dataset302_BraTSB/labelsTr -type f | wc -l

python fednnunet/run.py plan_and_preprocess "301 302" 3d_fullres --port 8080 -np 1 -npfp 1
"""


"""
python fednnunet/dataset_conversion/convert_brats_fed.py \
    -i /group/pmc079/jchin/data \
    --start_dataset_id 301 \
    --num_sites 2
"""