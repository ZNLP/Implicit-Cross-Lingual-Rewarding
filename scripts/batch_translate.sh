#!/bin/bash

#usage: batch_inference.sh <gpu_id> <type> <split>
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${1:-0}"


# Set the working directory for the project
export code_dir=/public/zhangjiajun/wyang/workspace/release/code/Implicit-Cross-Lingual-Rewarding
# Specify the Python environment
export python_env=/public/zhangjiajun/anaconda3/envs/qwq/bin/python


src=${2:-"en"}
target=${3:-"en"}
dataset=${4:-"ultrafeedback_binarized/subset/random_3000/"}
mode=${5:-"M0"}
checkpoint_dir=${6:-"/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"}


if [ ${src} = "all" ]; then
    src=("es" "ru" "de" "fr")
fi

if [ ${target} = "all" ]; then
    target=("es" "ru" "de" "fr")
fi



if echo "$checkpoint_dir" | grep -q "llama2"; then
    template="llama2"
fi
if echo "$checkpoint_dir" | grep -q "Llama-2"; then
    template="llama2"
fi
if echo "$checkpoint_dir" | grep -q "llama3"; then
    template="llama3"
fi
if echo "$checkpoint_dir" | grep -q "Llama-3-Instruct"; then
    template="llama3"
fi
if echo "$checkpoint_dir" | grep -q "Llama-3-Base"; then
    template="llama3_without_system_prompt"
fi
if echo "$checkpoint_dir" | grep -q "Qwen"; then
    template="qwen"
fi
if echo "$checkpoint_dir" | grep -q "Aya"; then
    template="aya"
fi
if echo "$checkpoint_dir" | grep -q "aya"; then
    template="aya"
fi
if echo "$checkpoint_dir" | grep -q "gemma"; then
    template="gemma2"
fi
if echo "$checkpoint_dir" | grep -q "mistral-7b-sft-beta"; then
    template="zephyr_align_with_DICE"
fi
if echo "$checkpoint_dir" | grep -q "zephyr-7b-beta"; then
    template="zephyr_align_with_DICE"
fi
if echo "$checkpoint_dir" | grep -q "Mistral-7B-Base"; then
    template="mistral"
fi
if echo "$checkpoint_dir" | grep -q "mistral-7b-base"; then
    template="mistral"
fi



question_path=${code_dir}/process_data/inference/${mode}/${checkpoint_dir##*/}/multilingual_generate/${dataset}
save_dir=${code_dir}/process_data/translate/${mode}/${checkpoint_dir##*/}/multilingual_generate/${dataset}

temperature=0

${python_env} -u utils/batch_translate.py \
    --question_path ${question_path} \
    --save_dir ${save_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --temperature ${temperature} \
    --max_tokens 8192 \
    --src_langs ${src[@]} \
    --tgt_langs ${target[@]} \
    --seed 42 \
    --template ${template} 