set -e

export CUDA_VISIBLE_DEVICES=3

# if the preprocessed dataset for Dataset100_Heart does not exist, run preprocessing
if [ ! -d "./nnUNet_preprocessed/Dataset100_Heart" ]; then
    nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
fi

nnUNetv2_train 100 3d_fullres 3 -device 'cuda'