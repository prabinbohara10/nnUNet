## End-to-end setup to run BraTS 2025 SSA dataset in nnUNet
```bash
mkdir nnUNetFrame
cd nnUNetFrame

# Create dataset folders
mkdir dataset
cd dataset
mkdir nnUNet_raw nnUNet_preprocessed nnUNet_results

python -m venv venv
source venv/bin/activate

git clone https://github.com/prabinbohara10/nnUNet.git
cd nnUNet
pip install -e .

# Optional
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

cd .. # Be inside the nnUNetFrame directory

# env_setup:
# For this env_setup.sh script can be directly run
export ORIGINAL_TRAIN_DATA="/home/azureuser/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
export ORIGINAL_VAL_ONLY_DATA="/home/azureuser/data/BraTS2024-SSA-Challenge-ValidationData" # This is used just to run the prediction/ inference

export nnUNet_raw=... # Absolute Path to nnUNet_raw folder
export nnUNet_preprocessed=... # Absolute Path to nnUNet_preprocessed folder
export nnUNet_results=... # Absolute Path to nnUNet_results folder


# Steps to run the actual scripts:

# 1. For training: Run python Dataset1137_BraTS23.py
python nnUNet/nnunetv2/dataset_conversion/Dataset1137_BraTS23.py # This formats the data as per the nnUNet and bring all the data to nnUNet_raw folder

# 2. 
nnUNetv2_plan_and_preprocess -d 1137 --verify_dataset_integrity # This is equivalent to running nnUNetv2_extract_fingerprint, nnUNetv2_plan_experiment and nnUNetv2_preprocess (in that order).
# Puts the output in nnUNet_preprocessed

# 3. 
nnUNetv2_train 1137 3d_fullres 0 -tr nnUNetTrainer_5epochs


```