import csv
import os
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.dataset_conversion.Dataset027_ACDC import make_out_dirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_preprocessed
from sklearn.model_selection import StratifiedKFold


def generate_MnM3(
    dataset_path: Path,
    csv_file: str,
    centre_id: int = None,
    save_splits=True,
    image_reader="NibabelIO",
    base_dataset_id=300,
):
    # Get patient ids from the csv file
    patient_ids = []
    patient_info = {}
    corrupted_samples = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        headers = [h.strip().lstrip("\ufeff") for h in next(reader)]
        print("CSV headers:", headers)

        if "ID" in headers:
            patient_index = headers.index("ID")
        elif "External code" in headers:
            patient_index = headers.index("External code")
        else:
            raise ValueError(f"Could not find patient ID column in CSV headers: {headers}")

        centre_index = headers.index("Centre")
        for row in reader:
            patient_id = row[patient_index]
            patient_id = str(patient_id).zfill(3)
            patient_dir = dataset_path / patient_id
            if not patient_dir.exists():
                continue

            ed_index = headers.index("ED")
            es_index = headers.index("ES")

            patient_info[patient_id] = {
                "centre": row[centre_index],
                "ed_frame": int(row[ed_index]),
                "es_frame": int(row[es_index]),
            }
            patient_ids.append(patient_id)

            corrupted_files = False

            # Case 1: ED/ES-style files
            if (
                (patient_dir / f"{patient_id}_ED.nii.gz").exists()
                or (patient_dir / f"{patient_id}_SA_ED.nii.gz").exists()
            ):
                patient_info[patient_id]["mode"] = "phases"
                suffixes = ["ED", "ES", "ED_gt", "ES_gt"]

                for suffix in suffixes:
                    if (patient_dir / f"{patient_id}_{suffix}.nii.gz").exists():
                        patient_info[patient_id][suffix] = (
                            patient_dir / f"{patient_id}_{suffix}.nii.gz"
                        )
                    elif (patient_dir / f"{patient_id}_SA_{suffix}.nii.gz").exists():
                        patient_info[patient_id][suffix] = (
                            patient_dir / f"{patient_id}_SA_{suffix}.nii.gz"
                        )
                    else:
                        corrupted_files = True
                        print(f"Patient {patient_id} is missing {suffix}.nii.gz")
                        break

                # Check readability
                for key in ["ED", "ES", "ED_gt", "ES_gt"]:
                    if corrupted_files:
                        break
                    try:
                        if image_reader == "NibabelIO":
                            nib.load(patient_info[patient_id][key])
                        elif image_reader == "SimpleITKIO":
                            sitk.ReadImage(str(patient_info[patient_id][key]))
                        else:
                            raise ValueError(f"Unsupported image reader: {image_reader}")
                    except Exception as e:
                        corrupted_files = True
                        print(
                            f"File {patient_info[patient_id][key]} has unsupported format: {e} for {image_reader} reader."
                        )

            # Case 2: single-SA-style cine files -> extract ED/ES frames
            elif (
                (patient_dir / f"{patient_id}_sa.nii.gz").exists()
                and (patient_dir / f"{patient_id}_sa_gt.nii.gz").exists()
            ):
                patient_info[patient_id]["mode"] = "phases_from_sa"
                patient_info[patient_id]["SA"] = patient_dir / f"{patient_id}_sa.nii.gz"
                patient_info[patient_id]["SA_gt"] = patient_dir / f"{patient_id}_sa_gt.nii.gz"

                try:
                    img = nib.load(patient_info[patient_id]["SA"])
                    gt = nib.load(patient_info[patient_id]["SA_gt"])

                    img_data = img.get_fdata()
                    gt_data = gt.get_fdata()

                    if img_data.ndim != 4 or gt_data.ndim != 4:
                        raise ValueError(
                            f"Expected 4D cine volumes for {patient_id}, got "
                            f"image ndim={img_data.ndim}, gt ndim={gt_data.ndim}"
                        )

                    ed = patient_info[patient_id]["ed_frame"]
                    es = patient_info[patient_id]["es_frame"]

                    if ed < 0 or ed >= img_data.shape[3]:
                        raise ValueError(f"ED frame {ed} out of bounds for {patient_id}")
                    if es < 0 or es >= img_data.shape[3]:
                        raise ValueError(f"ES frame {es} out of bounds for {patient_id}")

                    patient_info[patient_id]["ED_data"] = img_data[..., ed]
                    patient_info[patient_id]["ES_data"] = img_data[..., es]
                    patient_info[patient_id]["ED_gt_data"] = gt_data[..., ed]
                    patient_info[patient_id]["ES_gt_data"] = gt_data[..., es]
                    patient_info[patient_id]["affine"] = img.affine

                except Exception as e:
                    corrupted_files = True
                    print(
                        f"Patient {patient_id} SA cine extraction failed: {e}"
                    )

            else:
                corrupted_files = True
                print(f"Patient {patient_id} does not have supported image/label files.")

            if corrupted_files:
                # Remove patient from dictionary
                patient_info.pop(patient_id)
                patient_ids.remove(patient_id)
                corrupted_samples.append(patient_id)

    if corrupted_samples:
        print(
            f"{len(corrupted_samples)} patients: {corrupted_samples} have unsupported file format."
        )

    # print("Done checking files.")

    def patient_to_sample_names(pid):
        mode = patient_info[pid].get("mode")
        if mode in ["phases", "phases_from_sa"]:
            return [f"{pid}_ED", f"{pid}_ES"]
        else:
            raise ValueError(f"Patient {pid} has no valid mode. Entry: {patient_info[pid]}")
        
    # Perform stratified 5-fold cross validation based on the centres
    centre_folds = {}
    patient_ids = list(patient_info.keys())
    patient_centres = [patient_info[patient_id]["centre"] for patient_id in patient_ids]

    # Use as many folds as the smallest centre can support, capped at 5
    from collections import Counter
    centre_counts = Counter(patient_centres)
    min_class_count = min(centre_counts.values())

    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError(
            f"Not enough samples per centre for cross-validation. "
            f"Centre counts: {dict(centre_counts)}"
        )

    print(f"Using {n_splits}-fold CV for subset smoke test. Centre counts: {dict(centre_counts)}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(
        skf.split(patient_ids, patient_centres)
    ):
        train_patients = [patient_ids[i] for i in train_index]
        test_patients = [patient_ids[i] for i in test_index]

        if centre_id is not None:
            train_patients = [
                patient_id
                for patient_id in train_patients
                if patient_info[patient_id]["centre"] == str(centre_id)
            ]
            test_patients = [
                patient_id
                for patient_id in test_patients
                if patient_info[patient_id]["centre"] == str(centre_id)
            ]

        train_file_names = [
            name for patient_id in train_patients for name in patient_to_sample_names(patient_id)
        ]
        test_file_names = [
            name for patient_id in test_patients for name in patient_to_sample_names(patient_id)
        ]
        split_dict = {"train": train_file_names, "val": test_file_names}

        centre_folds[fold] = {
            "train": train_patients,
            "test": test_patients,
            "split_dict": split_dict,
        }

    if centre_id:
        dataset_id = base_dataset_id + centre_id
    else:
        dataset_id = base_dataset_id

    task_name = "MNMs3_fed" if centre_id else "MNMs3_centralized"
    out_dir, out_train_dir, out_labels_dir, out_test_dir = make_out_dirs(
        dataset_id, task_name=task_name
    )

    def save_subset(patient_ids, out_samples_dir):
        for patient_id in patient_ids:
            mode = patient_info[patient_id]["mode"]

            if mode == "phases":
                for suffix in ["ED", "ES"]:
                    sample_src = patient_info[patient_id][suffix]
                    sample_dst = out_samples_dir / f"{patient_id}_{suffix}_0000.nii.gz"
                    gt_src = patient_info[patient_id][f"{suffix}_gt"]
                    gt_dst = out_labels_dir / f"{patient_id}_{suffix}.nii.gz"
                    os.system(f"cp {sample_src} {sample_dst}")
                    os.system(f"cp {gt_src} {gt_dst}")

            elif mode == "phases_from_sa":
                affine = patient_info[patient_id]["affine"]

                for suffix, img_key, gt_key in [
                    ("ED", "ED_data", "ED_gt_data"),
                    ("ES", "ES_data", "ES_gt_data"),
                ]:
                    sample_dst = out_samples_dir / f"{patient_id}_{suffix}_0000.nii.gz"
                    gt_dst = out_labels_dir / f"{patient_id}_{suffix}.nii.gz"

                    nib.save(
                        nib.Nifti1Image(patient_info[patient_id][img_key], affine),
                        str(sample_dst),
                    )
                    nib.save(
                        nib.Nifti1Image(patient_info[patient_id][gt_key], affine),
                        str(gt_dst),
                    )

            else:
                raise ValueError(
                    f"Unsupported mode {mode} for patient {patient_id}"
                )

    # Build the full patient list that should physically exist in this dataset
    if centre_id is not None:
        dataset_patients = [
            patient_id
            for patient_id in patient_info.keys()
            if patient_info[patient_id]["centre"] == str(centre_id)
        ]
    else:
        dataset_patients = list(patient_info.keys())

    if save_splits:
        # For nnU-Net CV datasets, all samples go into training
        save_subset(dataset_patients, out_train_dir)

        splits_list = [fold_split["split_dict"] for fold_split in centre_folds.values()]
        preprocessed_dir = (
            nnUNet_preprocessed.replace('"', "") + f"/Dataset{dataset_id}_{task_name}/"
        )
        os.makedirs(preprocessed_dir, exist_ok=True)
        save_json(splits_list, preprocessed_dir + "splits_final.json")
        print(
            f"Saved splits for dataset {dataset_id} in {preprocessed_dir}splits_final.json"
        )

        num_training_cases = sum(
            2 if patient_info[pid]["mode"] in ["phases", "phases_from_sa"] else 1
            for pid in dataset_patients
        )
    else:
        # Fallback mode without splits: use the first fold only
        first_fold = centre_folds[0]
        save_subset(first_fold["train"], out_train_dir)
        save_subset(first_fold["test"], out_test_dir)
        num_training_cases = sum(
            2 if patient_info[pid]["mode"] in ["phases", "phases_from_sa"] else 1
            for pid in first_fold["train"]
        )

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={"background": 0, "LVBP": 1, "LVM": 2, "RV": 3},
        file_ending=".nii.gz",
        overwrite_image_reader_writer=image_reader,
        num_training_cases=num_training_cases,
    )
    
    print(
        f"Prepared dataset for centre {centre_id} with {len(dataset_patients)} patients"
    )


if __name__ == "__main__":
    import argparse

    class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=RawTextArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="MNMs conversion utility helper. This script can be used to convert MNMs data into the expected nnUNet "
        "format. It can also be used to create additional custom splits, for explicitly training on combinations "
        "of vendors A and B (see `--custom-splits`).\n"
        "If you wish to generate the custom splits, run the following pipeline:\n\n"
        "(1) Run `Dataset114_MNMs -i <raw_Data_dir>\n"
        "(2) Run `nnUNetv2_plan_and_preprocess -d 114 --verify_dataset_integrity`\n"
        "(3) Start training, but stop after initial splits are created: `nnUNetv2_train 114 2d 0`\n"
        "(4) Re-run `Dataset114_MNMs`, with `-s True`.\n"
        "(5) Re-run training.\n",
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="/data/MMs/",
        help="The downloaded MNMs dataset dir. Should contain a csv file, as well as Training, Validation and Testing "
        "folders.",
    )
    parser.add_argument(
        "-c",
        "--csv_file_name",
        type=str,
        default="211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        help="The csv file containing the dataset information.",
    ),
    parser.add_argument(
        "-d", "--dataset_id", type=int, default=114, help="nnUNet Dataset ID."
    )
    parser.add_argument(
        "-s",
        "--save_splits",
        type=bool,
        default=True,
        help="Save splits in nnUNet preprocessed directory.",
    )

    parser.add_argument(
        "--centre_id",
        nargs="+",
        type=int,
        default=None,
        help="Populate dataset for selected data centre. Accepts multiple values to create separate datasets. If not specified, the entire centralized dataset is populated. ",
    )

    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)
    if not args.centre_id:
        centre_ids = [None]
    else:
        centre_ids = set(args.centre_id)

    for centre_id in centre_ids:
        print(f"Populating dataset for centre {centre_id}")
        generate_MnM3(
            args.input_folder,
            args.csv_file_name,
            centre_id=centre_id,
            save_splits=args.save_splits,
            base_dataset_id=args.dataset_id,
        )

    print("Done!")
