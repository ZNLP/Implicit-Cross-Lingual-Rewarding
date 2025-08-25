#!/bin/bash

# ==============================================
# Generate Preference Pairs Script
# ==============================================
# Usage:
#   bash generate_preference_pair.sh <langs> <mode> <dpo_model> <ref_model> <data_type> <dataset> <upper_bound> <tmp_and_topp>
# Example:
#   bash generate_preference_pair.sh "en_es_ru_de_fr" "M0" "Llama-3-Base-8B-SFT-DPO" "Llama-3-Base-8B-SFT" "multilingual_generate_crosslingual_rewarding" "ultrafeedback_binarized/subset/random_3000/" "inf" "temp_0.9_top-p_1.0"
# Defaults:
#   - code_dir: "/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding"
#   - langs: "en_es_ru_de_fr"
#   - mode: "M0"
#   - dpo_model: "Llama-3-Base-8B-SFT-DPO"
#   - ref_model: "Llama-3-Base-8B-SFT"
#   - data_type: "multilingual_generate_crosslingual_rewarding"
#   - dataset: "ultrafeedback_binarized/subset/random_3000/"
#   - upper_bound: "inf"
#   - tmp_and_topp: "temp_0.9_top-p_1.0"
# ==============================================

# Specify the Python environment
export python_env=/public/zhangjiajun/anaconda3/envs/qwq/bin/python

# Define parameters
code_dir=${1:-"/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding"}

# Parse langs as an array using `_` as the separator
IFS='_' read -r -a langs <<< "${2:-"en_es"}"  # Correctly split string into an array

mode=${3:-"M0"}  # Training mode
dpo_model=${4:-"Llama-3-Base-8B-SFT-DPO"}  # DPO-trained model
ref_model=${5:-"Llama-3-Base-8B-SFT"}  # Reference model
data_type=${6:-"multilingual_generate_crosslingual_rewarding"}  # Data type for generation
dataset=${7:-"ultrafeedback_binarized/subset/random_3000/"}  # Dataset path
upper_bound=${8:-"inf"}  # Upper bound on preference pairs
tmp_and_topp=${9:-"temp_0.9_top-p_1.0"}  # Sampling parameters

# ==============================================
# Run Preference Pair Generation
# ==============================================
echo "=========================================="
echo "Starting Preference Pair Generation..."
echo "  - Code Dir: ${code_dir}"
echo "  - Target Languages: ${langs[*]}"  # Print all elements of langs array
echo "  - Experiment Mode: ${mode}"
echo "  - DPO Model: ${dpo_model}"
echo "  - Reference Model: ${ref_model}"
echo "  - Data Type: ${data_type}"
echo "  - Dataset Path: ${dataset}"
echo "  - Upper Bound: ${upper_bound}"
echo "  - Sampling Settings: ${tmp_and_topp}"
echo "=========================================="

${python_env} -u utils/generate_preference_pair.py \
    --code_dir "${code_dir}" \
    --langs "${langs[@]}" \
    --mode "${mode}" \
    --dpo_model "${dpo_model}" \
    --ref_model "${ref_model}" \
    --data_type "${data_type}" \
    --dataset "${dataset}" \
    --upper_bound "${upper_bound}" \
    --tmp_and_topp "${tmp_and_topp}" \
    --length_control

echo "Preference Pair Generation completed successfully!"
