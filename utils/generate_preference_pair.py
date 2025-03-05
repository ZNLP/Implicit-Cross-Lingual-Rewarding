import json
import os
import numpy as np
import pandas as pd
from copy import deepcopy

from utils import load_jsonl, write_jsonl

# Function to load response log probabilities from a JSONL file
def load_response_log_probs(file_path, key="response_log_probs"):
    log_probs_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if key in data:
                # Each entry in the list is now a list of log probabilities
                log_probs_list.append(data[key])
    return log_probs_list

def load_response_length(file_path, key="response"):
    responses_length = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if key in data:
                # Each response's length is a list of lengths of each item in the response
                response_length = [len(response) for response in data[key]]
                responses_length.append(response_length)
    return responses_length

# Function to calculate implicit reward scores
def calculate_implicit_reward(dpo_log_probs, ref_log_probs):
    # Since each dpo_log_probs and ref_log_probs is a list of lists, we process them element by element
    return [
        [dpo - ref for dpo, ref in zip(dpo_sublist, ref_sublist)]
        for dpo_sublist, ref_sublist in zip(dpo_log_probs, ref_log_probs)
    ]

# Function to adjust reward scores with length control
def add_lc_score(scores, lengths, alpha):
    # Assuming scores and lengths are both lists of lists, we process each sublist
    return [
        [score - alpha * length for score, length in zip(score_sublist, length_sublist)]
        for score_sublist, length_sublist in zip(scores, lengths)
    ]

# Function to optimize alpha
def optimize_alpha(scores, lengths, alpha_range=(0.0, 1.0, 0.001)):
    best_alpha = 0.0
    min_avg_diff = float('inf')
    results = []

    for alpha in np.arange(*alpha_range):
        # Adjust the scores using alpha
        adjusted_scores = add_lc_score(scores, lengths, alpha)

        # import pdb; pdb.set_trace()

        # Find the index of the highest and lowest LC scores
        chosen_idx_list = [np.argmax(sublist) for sublist in adjusted_scores]  # Index of max LC score
        reject_idx_list = [np.argmin(sublist) for sublist in adjusted_scores]  # Index of min LC score

        # Calculate the length difference for the highest and lowest LC scores
        chosen_lengths_list = [length[idx] for idx, length in zip(chosen_idx_list, lengths)]
        reject_lengths_list = [length[idx] for idx, length in zip(reject_idx_list, lengths)]

        # Calculate the average abs length difference
        avg_length_diff = np.mean([
            chosen_length - reject_length for chosen_length, reject_length in zip(chosen_lengths_list, reject_lengths_list)
        ])


        
        # Store the results
        results.append((alpha, avg_length_diff))

        # Update best_alpha if the average length difference is smaller
        if abs(avg_length_diff) < abs(min_avg_diff):
            min_avg_diff = avg_length_diff
            best_alpha = alpha

    return best_alpha, results

def generate_lc_dpo_pair(parallel_language_prompt_file, dpo_log_probs_file, lc_dpo_file, scores, lengths, alpha, upper_bound=float('inf')):
    adjusted_scores = add_lc_score(scores, lengths, alpha)

    chosen_idx_list = [np.argmax([score if score < upper_bound else -float('inf') for score in sublist]) for sublist in adjusted_scores]


    reject_idx_list = [np.argmin(sublist) for sublist in adjusted_scores]  # Index of min LC score

   
    parallel_language_prompt = load_jsonl(parallel_language_prompt_file)
    data = load_jsonl(dpo_log_probs_file)
    
    lc_dpo_pairs = []
    for idx, (chosen_idx, reject_idx) in enumerate(zip(chosen_idx_list, reject_idx_list)):
        lc_dpo_pairs.append({
            "prompt": parallel_language_prompt[idx]["instruction"],
            "chosen": data[idx]["response"][chosen_idx],
            "reject": data[idx]["response"][reject_idx],
            "chosen_score": adjusted_scores[idx][chosen_idx],
            "reject_score": adjusted_scores[idx][reject_idx],
            "id": data[idx]["instruction_id"] if "instruction_id" in data[idx] else idx
        })

    write_jsonl(lc_dpo_file, lc_dpo_pairs)

    return lc_dpo_pairs

# Main function to execute the process
def main(args):
    # File paths

    all_lc_dpo_pairs = []


    for lang in args.langs:
        print(f"Processing {lang}...upper bound {args.upper_bound}...")

        if "translateToEn_rewarding" in args.data_type and lang == "en":
            continue

        if args.mode == "M0":

            dpo_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.dpo_model}/{args.data_type}/{args.dataset}/{lang}.response_log_probs.jsonl.prediction.with_{args.dpo_model}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_log_probs.jsonl"
            ref_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.ref_model}/{args.data_type}/{args.dataset}/{lang}.response_log_probs.jsonl.prediction.with_{args.dpo_model}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_log_probs.jsonl"

            if "translateToEn_rewarding" in args.data_type:
                dpo_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.dpo_model}/{args.data_type}/{args.dataset}/{lang}.response_translate_to_en_log_probs.jsonl.prediction.with_{args.dpo_model}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_translate_to_en_log_probs.jsonl.to_en.response_translate_to_en_log_probs.jsonl"
                ref_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.ref_model}/{args.data_type}/{args.dataset}/{lang}.response_translate_to_en_log_probs.jsonl.prediction.with_{args.dpo_model}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_translate_to_en_log_probs.jsonl.to_en.response_translate_to_en_log_probs.jsonl"

        if args.mode == "M1":
            dpo_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.dpo_model}/{args.data_type}/{args.dataset}/{lang}.response_log_probs.jsonl.prediction.with_{args.mode}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_log_probs.jsonl"
            ref_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.ref_model}/{args.data_type}/{args.dataset}/{lang}.response_log_probs.jsonl.prediction.with_{args.mode}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_log_probs.jsonl"

            if "translateToEn_rewarding" in args.data_type:
                dpo_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.dpo_model}/{args.data_type}/{args.dataset}/{lang}.response_translate_to_en_log_probs.jsonl.prediction.with_{args.mode}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_translate_to_en_log_probs.jsonl.to_en.response_translate_to_en_log_probs.jsonl"
                ref_log_probs_file = f"{args.code_dir}/process_data/reward_score/{args.mode}/{args.ref_model}/{args.data_type}/{args.dataset}/{lang}.response_translate_to_en_log_probs.jsonl.prediction.with_{args.mode}.{args.tmp_and_topp}_top-k_-1_n-sample_10_max-tokens_2400_seed_42.to_{lang}.response_translate_to_en_log_probs.jsonl.to_en.response_translate_to_en_log_probs.jsonl"
        
        # Load data

        if "translateToEn_rewarding" in args.data_type:
            key="response_translate_to_en_log_probs"
        else:
            key="response_log_probs"

        dpo_log_probs = load_response_log_probs(dpo_log_probs_file, key)
        ref_log_probs = load_response_log_probs(ref_log_probs_file, key)


        # Ensure both lists have the same length
        if len(dpo_log_probs) != len(ref_log_probs):
            raise ValueError("Mismatch in the number of responses between DPO and reference log probs.")

        # Calculate implicit reward scores
        implicit_reward_scores = calculate_implicit_reward(dpo_log_probs, ref_log_probs)

        # Calculate response lengths
        response_lengths = load_response_length(dpo_log_probs_file)

        if args.length_control:
            
            # Optimize alpha
            best_alpha, optimization_results = optimize_alpha(implicit_reward_scores, response_lengths)

            # Save jsonl file
            lc_file = f"{args.code_dir}/process_data/length_control/{args.mode}/{args.dpo_model}_and_{args.ref_model}/{args.data_type}/{args.dataset}/{lang}.jsonl"
            if not os.path.exists(lc_file):
                os.makedirs(os.path.dirname(lc_file), exist_ok=True)

            write_jsonl(lc_file, optimization_results)




            print(f"Best alpha: {best_alpha}")

            lc_dpo_file = f"{args.code_dir}/data/{args.dataset.split('/')[0]}/{args.mode}/{args.dpo_model}_and_{args.ref_model}/{args.data_type}/{args.dataset.split('/')[1]}_{args.dataset.split('/')[2]}/length_control/{lang}_lc_alpha_{round(best_alpha, 3)}_dpo.jsonl"

            if args.upper_bound != float('inf'):
                lc_dpo_file = f"{args.code_dir}/data/{args.dataset.split('/')[0]}/{args.mode}/{args.dpo_model}_and_{args.ref_model}/{args.data_type}/{args.dataset.split('/')[1]}_{args.dataset.split('/')[2]}/length_control_upper_bound_{args.upper_bound}/{lang}_lc_alpha_{round(best_alpha, 3)}_dpo.jsonl"
        else:
            best_alpha = 0.0

            lc_dpo_file = f"{args.code_dir}/data/{args.dataset.split('/')[0]}/{args.mode}/{args.dpo_model}_and_{args.ref_model}/{args.data_type}/{args.dataset.split('/')[1]}_{args.dataset.split('/')[2]}/wo_length_control/{lang}_dpo.jsonl"


            if args.upper_bound != float('inf'):
                lc_dpo_file = f"{args.code_dir}/data/{args.dataset.split('/')[0]}/{args.mode}/{args.dpo_model}_and_{args.ref_model}/{args.data_type}/{args.dataset.split('/')[1]}_{args.dataset.split('/')[2]}/wo_length_control_upper_bound_{args.upper_bound}/{lang}_dpo.jsonl"


        # Generate DPO pairs
        parallel_language_prompt_file = f"{args.code_dir}/data/{args.dataset}/{lang}.jsonl"
        

        if not os.path.exists(lc_dpo_file):
            os.makedirs(os.path.dirname(lc_dpo_file), exist_ok=True)

        lc_dpo_pair = generate_lc_dpo_pair(parallel_language_prompt_file, dpo_log_probs_file, lc_dpo_file, implicit_reward_scores, response_lengths, best_alpha, args.upper_bound)

        all_lc_dpo_pairs.extend(lc_dpo_pair)


    all_lang = "_".join(args.langs)
    
    # Save all pairs
    all_lc_dpo_pairs_file = os.path.dirname(lc_dpo_file) + f"/{all_lang}_dpo.jsonl"

    write_jsonl(all_lc_dpo_pairs_file, all_lc_dpo_pairs)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO pairs with length control")
    
    parser.add_argument('--code_dir', type=str, default="/home/wyang/code/Implicit-Cross-Lingual-Rewarding", help="Code directory")
    parser.add_argument('--langs', type=str, nargs="+", default=["en"], help="language of the question")
    parser.add_argument('--mode', type=str, default="M0", help="mode of the model")
    parser.add_argument('--dpo_model', type=str, default="Llama-3-Base-8B-SFT-DPO", help="DPO model")
    parser.add_argument('--ref_model', type=str, default="Llama-3-Base-8B-SFT", help="Reference model")
    parser.add_argument('--data_type', type=str, default="multilingual_generate_crosslingual_rewarding", help="Data type")
    parser.add_argument('--dataset', type=str, default="ultrafeedback_binarized/subset/random_3000/", help="Dataset")
    parser.add_argument('--upper_bound', type=float, default=float('inf'), help="Upper bound")
    parser.add_argument('--tmp_and_topp', type=str, default="temp_0.9_top-p_1.0", help="Temp and top-p")
    
    parser.add_argument("--length_control", action="store_true", help="Whether to apply length control")

    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    main(args)