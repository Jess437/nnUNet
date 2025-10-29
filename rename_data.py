import os

if __name__ == "__main__":
    train_image_path = "./nnUNet_raw/Dataset100_Heart/imagesTr"
    train_label_path = "./nnUNet_raw/Dataset100_Heart/labelsTr"
    test_image_path = "./nnUNet_raw/Dataset100_Heart/imagesTs"
    prediction_path = "./nnUNet_raw/Dataset100_Heart/predict_pp"

    # Rename training images: patient0001.nii.gz -> patient_001_0000.nii.gz
    for filename in os.listdir(train_image_path):
        if filename.endswith(".nii.gz"):
            # Skip if already in target format
            if "_0000.nii.gz" in filename:
                continue
            base, ext = os.path.splitext(filename)
            base, ext2 = os.path.splitext(base)  # to handle .nii.gz
            patient_id = base.replace("patient", "")
            new_filename = f"patient_{int(patient_id):03d}_0000{ext2}{ext}"
            os.rename(os.path.join(train_image_path, filename),
                      os.path.join(train_image_path, new_filename))
            
    # Rename training labels: patient0001_gt.nii.gz -> patient_001.nii.gz
    for filename in os.listdir(train_label_path):
        if filename.endswith(".nii.gz"):
            # Skip if already in target format (no _gt in filename and has underscore pattern)
            if "_gt" not in filename and filename.startswith("patient_"):
                continue
            base, ext = os.path.splitext(filename)
            base, ext2 = os.path.splitext(base)  # to handle .nii.gz
            patient_id = base.replace("patient", "").replace("_gt", "")
            new_filename = f"patient_{int(patient_id):03d}{ext2}{ext}"
            os.rename(os.path.join(train_label_path, filename),
                      os.path.join(train_label_path, new_filename))
            
    # Rename test images: patient0001.nii.gz -> patient_001_0000.nii.gz
    for filename in os.listdir(test_image_path):
        if filename.endswith(".nii.gz"):
            # Skip if already in target format
            if "_0000.nii.gz" in filename:
                continue
            base, ext = os.path.splitext(filename)
            base, ext2 = os.path.splitext(base)  # to handle .nii.gz
            patient_id = base.replace("patient", "")
            new_filename = f"patient_{int(patient_id):03d}_0000{ext2}{ext}"
            os.rename(os.path.join(test_image_path, filename),
                      os.path.join(test_image_path, new_filename))

    # Rename predictions: patient_051.nii.gz -> patient0051_predict.nii.gz
    if os.path.exists(prediction_path):
        for filename in os.listdir(prediction_path):
            if filename.endswith(".nii.gz"):
                # Skip if already in target format
                if "_predict" in filename:
                    continue
                base, ext = os.path.splitext(filename)
                base, ext2 = os.path.splitext(base)  # to handle .nii.gz
                patient_id = base.replace("patient_", "")
                new_filename = f"patient{int(patient_id):04d}_predict{ext2}{ext}"
                os.rename(os.path.join(prediction_path, filename),
                          os.path.join(prediction_path, new_filename))
            
    print("Renaming completed.")