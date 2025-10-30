import nibabel as nib
from pathlib import Path
import pandas as pd
import numpy as np
import os

def analyze_nifti_dataset(nii_files: list, output_csv: str):
    records = []
    for nii_path in nii_files:
        try:
            img = nib.load(str(nii_path))
            header = img.header
            shape = header.get_data_shape()
            zooms = header.get_zooms()[:3]  # voxel spacing
            voxel_count = shape[0] * shape[1] * shape[2]
            affine = img.affine
            
            # Load actual data to count voxels per label
            data = img.get_fdata()
            unique_labels, label_counts = np.unique(data, return_counts=True)
            
            # Create label count string
            label_info = "; ".join([f"label_{int(label)}={count}" for label, count in zip(unique_labels, label_counts)])
            
            # Print detailed info for each file
            print(f"\n{nii_path.name}:")
            for label, count in zip(unique_labels, label_counts):
                print(f"  Label {int(label)}: {count} voxels")

            record = {
                "file": nii_path.name,
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],
                "spacing_x(mm)": round(zooms[0], 4),
                "spacing_y(mm)": round(zooms[1], 4),
                "spacing_z(mm)": round(zooms[2], 4),
                "voxel_count": voxel_count,
                "qoffset_x(mm)": round(affine[0, 3], 3),
                "qoffset_y(mm)": round(affine[1, 3], 3),
                "qoffset_z(mm)": round(affine[2, 3], 3),
                "label_voxel_counts": label_info,
            }
            records.append(record)
        except Exception as e:
            print(f"Can't read {nii_path.name}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nAnalyzed {len(df)} NIfTI files, results saved to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    train_image_path = "./nnUNet_raw/Dataset100_Heart/labelsTr"
    nii_files = [f for f in list(Path(train_image_path).rglob("*.nii*"))]
    nii_files = sorted(nii_files, key=lambda x: x.name)
    analyze_nifti_dataset(nii_files, output_csv="dataset_info.csv")