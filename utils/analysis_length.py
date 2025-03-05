from utils import load_jsonl, write_jsonl
import numpy as np
import os

from langdetect import detect

from pyfranc import franc

def char_len(text):
    return len(text)

def word_len(text):
    return len(text.split())

def token_len(text, tokenizer):
    return len(tokenizer(text))

def analysis_length(data, len_type="char_len", tokenizer=None):
    if len_type == "char_len":
        return [char_len(sample) for sample in data]
    elif len_type == "word_len":
        return [word_len(sample) for sample in data]
    elif len_type == "token_len":
        return [token_len(sample, tokenizer) for sample in data]
    else:
        raise ValueError("Invalid length type")

def compare_length(data1, data2, len_type="char_len"):
    lens1 = analysis_length(data1, len_type)
    lens2 = analysis_length(data2, len_type)

    data1_is_longer = [1 if l1 > l2 else 0 for l1, l2 in zip(lens1, lens2)]
    data2_is_longer = [1 if l1 < l2 else 0 for l1, l2 in zip(lens1, lens2)]

    print(f"Chosen is longer: {sum(data1_is_longer)/len(data1)}")
    
def detect_lang(text, lang):
    if lang == "zh":
        detect_lang = "zh-cn"
    else:
        detect_lang = lang  
    try:  
        if detect(text) != detect_lang:
            return False
        else:
            return True
    except:
        # raise ValueError(f"Error in detecting language for {text}")
        return False

def detect_lang_franc(text, lang):
    if lang == "zh":
        detect_lang = "zh-cn"
    else:
        detect_lang = lang  
    try:  
        if franc.lang_detect(response, minlength=1)[0][0] != detect_lang:
            return False
        else:
            return True
    except:
        # raise ValueError(f"Error in detecting language for {text}")
        return False

            

def main():
    
    data_dir = "/public/zhangjiajun/wyang/workspace/dual_dpo/dual_dpo_v15/results/M1/x-alpacaeval/Mistral-7B-Base-SFT-DPO_lr_1e-7_bs_1_ga_4_gpus_8_M0_dpo_reward_mistral-7b-base-sft-dpo_ultrafeedback_random_3000_5_langs_multilingual_generate_length_control_upper_bound_100_beta_0.1_ftx_1/multilingual_generate"

    print(f"Data dir: {data_dir}")
    for lang in ["en", "es", "ru", "fr", "de"]:
        
        
        ##### 1.Preference pair data #####
        if "wo_length_control" in data_dir:
            data_path = os.path.join(data_dir, f"{lang}_dpo.jsonl")
        else:
            import glob
            data_path = glob.glob(os.path.join(data_dir, f"{lang}_lc_alpha_*_dpo.jsonl"))[0]
        data = load_jsonl(data_path)

        chosen_data = [d["chosen"] for d in data]

        all_num = len(chosen_data)

        filter_chosen_data = [d for d in chosen_data if detect_lang(d, lang)]

        detected_num = len(filter_chosen_data)
        
        print(f"Language: {lang}")
        
        char_lens = analysis_length(chosen_data, len_type="char_len")
        print("Chosen data, char_len: mean: {}, max: {}, min: {}, detected_lang_ratio: {}".format(np.mean(char_lens), np.max(char_lens), np.min(char_lens), detected_num/all_num))

        rejected_data = [d.get("reject", d.get("rejected")) for d in data]

        char_lens = analysis_length(rejected_data, len_type="char_len")
        print("Rejected data, char_len: mean: {}, max: {}, min: {}".format(np.mean(char_lens), np.max(char_lens), np.min(char_lens)))

        compare_length(chosen_data, rejected_data, len_type="char_len")


        # ##### 2.Response data #####

        import glob

        if "crosslingual" in data_dir:
            data_path = glob.glob(os.path.join(data_dir, f"*_{lang}.jsonl"))[0] 
        else:
            data_path = glob.glob(os.path.join(data_dir, f"{lang}*.jsonl"))[0]
        data = load_jsonl(data_path)
        
        
        response_data = [r for d in data for r in d["response"]]

        all_num = len(response_data)

        response_data = [d for d in response_data if detect_lang(d, lang)]

        detected_num = len(response_data)

        char_lens = analysis_length(response_data, len_type="char_len")

        

        
        print(f"Language: {lang}")
        
        print("Response data, char_len: mean: {}, max: {}, min: {}, detected_lang_ratio: {}".format(np.mean(char_lens), np.max(char_lens), np.min(char_lens), detected_num/all_num))


    

    # tokenizer_path = "/public/zhangjiajun/PretrainModels/princeton-nlp/Llama-3-Base-8B-SFT-DPO"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # word_lens = analysis_length(data, len_type="word_len")
    # token_lens = analysis_length(data, len_type="token_len", tokenizer=tokenizer)
    # print("word_len: mean: {}, max: {}, min: {}".format(np.mean(word_lens), np.max(word_lens), np.min(word_lens)))
    # print("token_len: mean: {}, max: {}, min: {}".format(np.mean(token_lens), np.max(token_lens), np.min(token_lens))

if __name__ == "__main__":
    main()