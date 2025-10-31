set -e

export CUDA_VISIBLE_DEVICES=3
export dataset_id=100
export fold_id=0                        # (0, 1, 2, 3, 4)
export configuration=3d_fullres         # (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
export trainer=nnUNetTrainer            # refer to ./nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py to set desired epochs
export plan=nnUNetPlans

# if the preprocessed dataset for Dataset100_Heart does not exist, run preprocessing
if [ ! -d "./nnUNet_preprocessed/Dataset100_Heart" ]; then
    nnUNetv2_plan_and_preprocess -d $dataset_id --verify_dataset_integrity
fi

# print training start message
echo "=================================================================================="
echo "Training started"
echo "=================================================================================="

# training (note: validation in this step is only partial to save time)
nnUNetv2_train $dataset_id $configuration $fold_id -tr $trainer -p $plan

# print validation start message
echo "=================================================================================="
echo "Validation started"
echo "=================================================================================="

# complete validation
nnUNetv2_train $dataset_id $configuration $fold_id -tr $trainer -p $plan --val --val_best --npz

# print best inference command message
echo "=================================================================================="
echo "Best inference command generation started"
echo "=================================================================================="

# to generate best inference command
nnUNetv2_find_best_configuration $dataset_id -c $configuration -f $fold_id -tr $trainer -p $plan

# then go to ./nnUNet_results/Dataset100_Heart/inference_instructions.txt fetch the best inference commands (better)
# or you can just simply use below two commands
# "nnUNetv2_predict -d $dataset_id -i ./nnUNet_raw/Dataset100_Heart/imagesTs -o ./nnUNet_raw/Dataset100_Heart/predict -f $fold_id -tr $trainer -c $configuration -p $plan"
# "nnUNetv2_apply_postprocessing -i ./nnUNet_raw/Dataset100_Heart/predict -o ./nnUNet_raw/Dataset100_Heart/predict_pp -pp_pkl_file ./nnUNet_results/Dataset100_Heart/${trainer}__${plan}__${configuration}/crossval_results_folds_${fold_id}/postprocessing.pkl -np 8 -plans_json ./nnUNet_results/Dataset100_Heart/${trainer}__${plan}__${configuration}/crossval_results_folds_${fold_id}/plans.json"

# then use 
# "python rename_data.py"
# "zip -r ./predict.zip ./nnUNet_raw/Dataset100_Heart/predict_pp"