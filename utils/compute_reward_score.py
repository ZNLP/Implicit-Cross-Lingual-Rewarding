import os
import sys
import torch
import numpy as np
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from utils import load_jsonl, write_jsonl

from inference import SFT_PROMPT, EOS_TOKEN, PREFIX, LANGUAGES

def cross_implicit_reward_prompt(lang, question_key, template, d):
    if lang == "en":
        question = d[question_key]
    else:
        question = PREFIX.format(language=LANGUAGES[lang]) + d[question_key]

    prompt = SFT_PROMPT[template].format(instruction=question)
       
    if args.verbose:
        print(colored(question, "blue"))
    return prompt

def prepare_data(tokenizer, dataset, batch_size, eos_token, instruction_key, cal_key, cal_key_idx):

    tensor_dataset = []

    for i in tqdm(range(0, len(dataset), batch_size)):

        batch = dataset[i:i+batch_size]

        

        eot_id = tokenizer.encode(eos_token, add_special_tokens=False)

        if type(data[0][cal_key]) == str:
            inputs = tokenizer([sample[instruction_key] + sample[cal_key] + eos_token for sample in batch], 
                           return_tensors='pt', padding=True, add_special_tokens=False)
        else:
            inputs = tokenizer([sample[instruction_key] + sample[cal_key][cal_key_idx] + eos_token for sample in batch], 
                            return_tensors='pt', padding=True, add_special_tokens=False)


        ppl_mask = torch.ones_like(inputs['input_ids'])

        for idx, sample in enumerate(batch):

            inputs_response = tokenizer(sample[instruction_key], return_tensors='pt', padding=False, add_special_tokens=False)

            length = inputs_response['input_ids'].size(1)

            ppl_mask[idx, :length] = 0

            ppl_mask[idx, inputs['input_ids'][idx] == tokenizer.pad_token_id] = 0

            ppl_mask[idx, inputs['input_ids'][idx] == eot_id] = 0
        
        tensor_dataset.append({'inputs': inputs, 'ppl_mask': ppl_mask})

    return tensor_dataset


def compute_score(model, tensor_dataset, score_type):
    """
    Computes various scores (log-probabilities, perplexity, average loss, or loss) based on the given score_type.

    Args:
        model: The model to evaluate (should be compatible with Hugging Face).
        tensor_dataset: A dataset containing input tensors and masks.
        score_type: The type of score to compute ('log_pi', 'ppl', 'avg_loss', or 'loss').

    Returns:
        score_list: A list of scores for each sample in the dataset.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import numpy as np

    model.eval()
    score_list = []

    for batch in tqdm(tensor_dataset):
        
        batch['inputs'].to('cuda')
        batch_size = batch['inputs']['input_ids'].size(0)

        with torch.no_grad():
            outputs = model(**batch['inputs'], return_dict=True)

        logits = outputs.logits

        # Reshape logits to align with labels
        logits = logits.view(batch_size, -1, logits.size(-1))[:, :-1, :]
        labels = batch['inputs']['input_ids'][:, 1:]

        if score_type == "log_probs":
            # Compute log-probabilities for each token
            # import pdb; pdb.set_trace()
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).cpu()

            # Mask log-probabilities for tokens that are not in the response
            log_probs = log_probs * batch['ppl_mask'][:, :-1]

            # Sum log-probabilities for the entire sequence
            sequence_log_probs = log_probs.sum(dim=1).cpu().numpy()
            score_list.append(sequence_log_probs)

        else:
            # Compute loss for each token
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='none')
            loss = loss.reshape(batch_size, -1).cpu()

            # Mask the loss for the tokens that are not in the response
            loss = loss * batch['ppl_mask'][:, :-1]

            # Compute the average loss for each sample
            loss = loss.view(batch_size, -1).sum(dim=1)
            avg_loss = loss / batch['ppl_mask'].view(batch_size, -1).sum(dim=1)

            # Compute the perplexity for each sample
            ppl = torch.exp(avg_loss)

            if score_type == "ppl":
                score_list.append(ppl.numpy())
            elif score_type == "avg_loss":
                score_list.append(avg_loss.numpy())
            elif score_type == "loss":
                score_list.append(loss.numpy())

    score_list = np.concatenate(score_list).tolist()

    return score_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--template", type=str, default="llama3")
    parser.add_argument("--langs", nargs='+', required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_type", type=str, default="log_probs")
    parser.add_argument("--instruction_key", type=str, default="instruction")
    parser.add_argument("--cal_key", nargs='+', required=True)
    parser.add_argument("--sanity_check", action='store_true')

    parser.add_argument("--format_prompt", action='store_true')
    parser.add_argument("--rewarding_type", type=str, default="crosslingual_rewarding")


    args = parser.parse_args()


   

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # build model
    #model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='cuda')
    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenizer.padding_side = "right"  # Allow batched inference

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # import pdb; pdb.set_trace()

    if args.format_prompt or args.rewarding_type == "crosslingual_rewarding" or args.rewarding_type == "translateToEn_rewarding":
        en_data = load_jsonl(glob.glob(os.path.join(args.data_path, f"en*.jsonl"))[0])

        if "subset" in en_data[0]:
            en_data = {sample["subset"]+str(sample["id"]): sample for sample in en_data}
        else:
            en_data = {str(sample.get("instruction_id", sample.get("id"))): sample for sample in en_data}


    for cal_key in args.cal_key:
        for lang in args.langs:
            
            data_file = glob.glob(os.path.join(args.data_path, f"{lang}*.jsonl"))[0]


            output_file = os.path.join(args.output_path, f"{os.path.basename(data_file).replace('.jsonl', f'.{cal_key}_{args.score_type}.jsonl')}")

            print(f"Processing data for {lang}...")
            
            # load data
            data = load_jsonl(data_file)

            if args.sanity_check:
                data = data[0:1]

            if type(data[0][cal_key]) == str:
                cal_key_num = 1
            elif type(data[0][cal_key]) == list:
                cal_key_num = len(data[0][cal_key])
            else:
                raise ValueError("cal_key should be either str or list")

            for cal_key_idx in range(cal_key_num):

                # format prompt
                if args.format_prompt:
                    for sample in data:
                        if args.rewarding_type == "crosslingual_rewarding":
                            if "subset" in sample:
                                tag = sample["subset"]+str(sample["id"])
                            else:
                                tag = str(sample.get("instruction_id", sample.get("id")))

                            sample['format_prompt'] = cross_implicit_reward_prompt(lang, args.instruction_key, args.template, en_data[tag])
                        elif args.rewarding_type == "multilingual_rewarding":
                            sample['format_prompt'] = SFT_PROMPT[args.template].format(instruction=sample[args.instruction_key])
                        elif args.rewarding_type == "translateToEn_rewarding":
                            if "subset" in sample:
                                tag = sample["subset"]+str(sample["id"])
                            else:
                                tag = str(sample.get("instruction_id", sample.get("id")))
                            sample['format_prompt'] = SFT_PROMPT[args.template].format(instruction=en_data[tag][args.instruction_key])

                           

                        else:
                            raise ValueError("rewarding_type should be either crosslingual_rewarding or multilingual_rewarding")
                           
                # tokenize data
                if args.format_prompt:
                    tensor_dataset = prepare_data(tokenizer, data, args.batch_size, EOS_TOKEN[args.template], "format_prompt", cal_key, cal_key_idx)
                else:
                    tensor_dataset = prepare_data(tokenizer, data, args.batch_size, EOS_TOKEN[args.template], args.instruction_key, cal_key, cal_key_idx)

                # compute score for each data
                scores = compute_score(model, tensor_dataset, args.score_type)

                # add score to data
                for sample, score in zip(data, scores):
                    if f'{cal_key}_{args.score_type}' not in sample:
                        sample[f'{cal_key}_{args.score_type}'] = []
                    sample[f'{cal_key}_{args.score_type}'].append(np.round(score, 4))


            # save data
            write_jsonl(output_file, data)

        

        




