#!/bin/bash

# ==============================================
# Batch Inference Script
# ==============================================
# Usage:
#   bash batch_inference.sh <gpu_id> <type> <dataset> <mode> <checkpoint_dir>
# Example:
#   bash batch_inference.sh 0 "es" "ultrafeedback_binarized/subset/random_2000_6000/" "M0" "/path/to/checkpoint"
# Defaults:
#   - gpu_id: 0
#   - type: "en"
#   - dataset: "ultrafeedback_binarized/subset/random_2000_6000/"
#   - mode: "M0"
#   - checkpoint_dir: "/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"
# ==============================================

# Disable parallel tokenization to avoid issues
export TOKENIZERS_PARALLELISM=false

# Set CUDA device (default: GPU 0)
export CUDA_VISIBLE_DEVICES="${1:-0}"

# Inference parameters
type=${2:-"en"}  # Target language type
dataset=${3:-"ultrafeedback_binarized/subset/random_3000/"}  # Dataset path
mode=${4:-"M0"}  # Training mode
checkpoint_dir=${5:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"}  # Model checkpoint path

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
# Set Inference Parameters
# ==============================================
save_dir="${code_dir}/process_data/inference/${mode}/${checkpoint_dir##*/}/multilingual_generate/${dataset}"
temperature=0.9
top_p=1.0
sample_num=10
question_path="${code_dir}/data/${dataset}"

# ==============================================
# Run Batch Inference
# ==============================================
echo "=========================================="
echo "Starting batch inference..."
echo "  - CUDA Device: ${CUDA_VISIBLE_DEVICES}"
echo "  - Type (Language): ${type}"
echo "  - Dataset: ${dataset}"
echo "  - Mode: ${mode}"
echo "  - Checkpoint Directory: ${checkpoint_dir}"
echo "  - Template: ${template}"
echo "  - Save Directory: ${save_dir}"
echo "=========================================="

${python_env} -u utils/batch_inference.py \
    --question_path "${question_path}" \
    --question_key "instruction" \
    --save_dir "${save_dir}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --temperature "${temperature}" \
    --max_tokens 2400 \
    --langs "${type}" \
    --seed 42 \
    --template "${template}" \
    --n_generate_sample "${sample_num}" \
    --best_of "${sample_num}" \
    --top_p "${top_p}" \
    --mode "${mode}"

echo "Batch inference completed successfully!"
    