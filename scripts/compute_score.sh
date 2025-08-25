#!/bin/bash

# ==============================================
# Compute Reward Score Script
# ==============================================
# Usage:
#   bash compute_reward_score.sh <gpu_id> <languages> <rewarding_type> <ckpt_id> <dataset> <mode> <policy_model_dir> <ref_model_dir>
# Example:
#   bash compute_reward_score.sh 0 "es" "translateToEn_rewarding" 1 "alpagasus/subset/id_0_3000" "M0" "/path/to/policy_model" "/path/to/ref_model"
# Defaults:
#   - gpu_id: 0
#   - languages: "es"
#   - rewarding_type: "translateToEn_rewarding"
#   - ckpt_id: 0 (0: ref_model, 1: policy_model)
#   - dataset: "alpagasus/subset/id_0_3000"
#   - mode: "M0"
#   - policy_model_dir: "/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"
#   - ref_model_dir: "/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT"
# ==============================================

# Set the working directory for the project
export code_dir=/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding
# Specify the Python environment
export python_env=/public/zhangjiajun/anaconda3/envs/qwq/bin/python


# Set CUDA device (default: GPU 0)
export CUDA_VISIBLE_DEVICES=${1:-"0"}

# Define key parameters
languages=${2:-"es"}  # Target languages
rewarding_type=${3:-"translateToEn_rewarding"}  # Type of reward computation
ckpt_id=${4:-"0"}  # 0 = ref_model, 1 = policy_model
dataset=${5:-"ultrafeedback_binarized/subset/random_3000/"}  # Dataset path
mode=${6:-"M0"}  # Training mode
policy_model_dir=${7:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"}  # Policy model path
ref_model_dir=${8:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT"}  # Reference model path


# ==============================================
# Select Checkpoint Directory
# ==============================================
if [ ${ckpt_id} -eq 0 ]; then
    checkpoint_dir=${ref_model_dir}
elif [ ${ckpt_id} -eq 1 ]; then
    checkpoint_dir=${policy_model_dir}
else
    echo "Invalid ckpt_id: ${ckpt_id}. Must be 0 (ref_model) or 1 (policy_model)."
    exit 1
fi

# ==============================================
# Model-Specific Template Selection
# ==============================================
if echo "$checkpoint_dir" | grep -qi "llama2"; then
    template="llama2"
elif echo "$checkpoint_dir" | grep -qi "llama3"; then
    template="llama3"
elif echo "$checkpoint_dir" | grep -qi "Llama-3-Base"; then
    template="llama3_without_system_prompt"
elif echo "$checkpoint_dir" | grep -qi "Qwen"; then
    template="qwen"
elif echo "$checkpoint_dir" | grep -qi "Aya"; then
    template="aya"
elif echo "$checkpoint_dir" | grep -qi "Gemma"; then
    template="gemma2"
elif echo "$checkpoint_dir" | grep -qi "mistral-7b-sft-beta"; then
    template="zephyr_align_with_DICE"
elif echo "$checkpoint_dir" | grep -qi "zephyr-7b-beta"; then
    template="zephyr_align_with_DICE"
elif echo "$checkpoint_dir" | grep -qi "Mistral-7B-Base"; then
    template="mistral"
else
    echo "Warning: No matching model template found. Using default template."
    template="default"
fi

# ==============================================
# Define Reward Computation Parameters
# ==============================================
kwargs=""  # Additional script arguments
cal_key=("response")  # Default calculation key

data_path="${code_dir}/process_data/inference/${mode}/${policy_model_dir##*/}/multilingual_generate/${dataset}"

if [ "${rewarding_type}" = "crosslingual_rewarding" ]; then 
    output_path="${code_dir}/process_data/reward_score/${mode}/${checkpoint_dir##*/}/multilingual_generate_crosslingual_rewarding/${dataset}"
    instruction_key="cross_implicit_reward_prompt"

elif [ "${rewarding_type}" = "multilingual_rewarding" ]; then 
    output_path="${code_dir}/process_data/reward_score/${mode}/${checkpoint_dir##*/}/multilingual_generate_multilingual_rewarding/${dataset}"
    instruction_key="prompt"

elif [ "${rewarding_type}" = "translateToEn_rewarding" ]; then 
    data_path="${code_dir}/process_data/translate/${mode}/${policy_model_dir##*/}/multilingual_generate/${dataset}"
    output_path="${code_dir}/process_data/reward_score/${mode}/${checkpoint_dir##*/}/multilingual_generate_translateToEn_rewarding/${dataset}"
    instruction_key="instruction"
    kwargs="--format_prompt"
    cal_key=("response_translate_to_en")
else
    echo "Invalid rewarding_type: ${rewarding_type}. Exiting."
    exit 1
fi

score_type="log_probs"  # Scoring metric

# ==============================================
# Run Reward Score Computation
# ==============================================
echo "=========================================="
echo "Starting reward score computation..."
echo "  - CUDA Device: ${CUDA_VISIBLE_DEVICES}"
echo "  - Target Languages: ${languages}"
echo "  - Rewarding Type: ${rewarding_type}"
echo "  - Checkpoint Directory: ${checkpoint_dir}"
echo "  - Model Template: ${template}"
echo "  - Data Path: ${data_path}"
echo "  - Output Path: ${output_path}"
echo "=========================================="

${python_env} utils/compute_reward_score.py \
    --model_path "${checkpoint_dir}" \
    --template "${template}" \
    --data_path "${data_path}"  \
    --batch_size 3 \
    --langs "${languages}" \
    --output_path "${output_path}" \
    --instruction_key "${instruction_key}" \
    --cal_key "${cal_key[@]}" \
    --score_type "${score_type}" \
    --rewarding_type "${rewarding_type}" \
    ${kwargs}

echo "Reward score computation completed successfully!"
