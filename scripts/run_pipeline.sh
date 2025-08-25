#!/bin/bash


# Set the working directory for the project
export code_dir=/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding
# Specify the Python environment
export python_env=/public/zhangjiajun/anaconda3/envs/qwq/bin/python

# Set the default CUDA device (can be overridden via command-line arguments)
export CUDA_VISIBLE_DEVICES=${1:-"0"}

# Set the target languages (default: Spanish "es")
languages=${2:-"en"}

# Set the rewarding type (default: "crosslingual_rewarding")
# Available options:
#   - crosslingual_rewarding: Reward based on cross-lingual evaluation.
#   - multilingual_rewarding: Reward based on multilingual evaluation.
#   - translateToEn_rewarding: Reward based on translation to English.
rewarding_type=${3:-"crosslingual_rewarding"}

# Specify the dataset (default: "ultrafeedback_binarized/subset/random_3000/")
dataset=${4:-"ultrafeedback_binarized/subset/random_100/"}

mode=${5:-"M0"}  # Training mode
policy_model_dir=${6:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"}  # Policy model path
ref_model_dir=${7:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT"}  # Reference model path


# Print experiment settings
echo "=========================================="
echo "Experiment Configuration:"
echo "  - Rewarding type: ${rewarding_type}"
echo "  - Dataset: ${dataset}"
echo "  - Languages: ${languages}"
echo "  - CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Check if the "scripts" directory exists
if [ ! -d "scripts" ]; then
    echo "Error: 'scripts' directory not found!"
    exit 1
fi

# Run batch inference
echo "Running batch inference..."
sh scripts/batch_inference.sh ${CUDA_VISIBLE_DEVICES} ${languages} ${dataset} ${mode} ${policy_model_dir}

# Compute scores for multiple iterations
for iter in 0 1; do
    echo "Computing scores for iteration ${iter}..."
    sh scripts/compute_score.sh ${CUDA_VISIBLE_DEVICES} ${languages} ${rewarding_type} ${iter} ${dataset} ${mode} ${policy_model_dir} ${ref_model_dir}
done

echo "Experiment completed successfully!"
