from __future__ import annotations

import os
import time
import json
import argparse
from termcolor import colored
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from inference import LlamaInferce, STOP, LANGUAGES, PREFIX, SUFFIX, SFT_PROMPT
from transformers import AutoTokenizer

from utils import load_jsonl, load_json

TIMEOUT_SECONDS = 40

def batch(iterable, n=-1):
    l = len(iterable)
    if n < 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generator(solver, output):
    solver.get_llm_response(output)
    return solver

def batch_main(args, llm, sampling_params, questions):
        
    try:
        solvers = [LlamaInferce(args, question) for question in questions]

        prompts = []
        stop_words = []
        for solver in solvers:
            prompt_text = solver.get_llm_request()
            prompts.append(prompt_text)
            stop_words.extend(STOP[args.template])
        
        if len(solvers) < 1:
            raise ValueError("No solver found.")
            
        sampling_params.stop = list(set(stop_words))
        outputs = llm.generate(prompts, sampling_params)

        with ProcessPool(max_workers=min(len(solvers), os.cpu_count())) as pool:
            future = pool.map(generator, solvers, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()

        if len(solvers) > 100:  
            progress_bar = tqdm(total=len(solvers), desc="Execute")  
        else:  
            progress_bar = None 


        next_solvers = []
        while True:
            try:
                result = next(iterator)
                next_solvers.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                next_solvers.append(None)
                print(error)
            except Exception as error:
                print(error)
                next_solvers.append(None)
            if progress_bar is not None:
                progress_bar.update(1) 

        if progress_bar is not None:
            progress_bar.close() 

        solvers = next_solvers
        

    except Exception as e:
        print(colored(f"Exception: {e}", "red"))
        return [""] * len(questions)
    
    jsonlines = {}
    for solver in solvers:            
        try:
            response = solver.response
            prompt = solver.prompt
            jsonlines[solver.question] = {
                "prompt": prompt,
                "response": response,
            }
        except:
            raise ValueError("Error in response generation")
    return jsonlines

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-q', '--question_path', type=str, default=None, help="file path of question file")
    args.add_argument('--question_key', type=str, default="instruction", help="questioin key in json")
    args.add_argument('--save_dir', type=str, default=None, help="folder to save prediction file.")
    args.add_argument('--save_file', type=str, default=None, help="file to save prediction file.")
    args.add_argument('--num_per_inference', type=int, default=-1, help="number of questions per inference")

    args.add_argument('--xalpaca_eval', action="store_true", help="for x-alpacaeval")
    args.add_argument('--langs', type=str, nargs="+", default=["en"], help="language of the question")
    args.add_argument('--with_prefix', action="store_true", help="add prefix to the prompt")
    args.add_argument('--cross_gen', action="store_true", help="cross-lingual generation")
    
    # model
    args.add_argument('-c', '--checkpoint_dir', type=str, default=None, help="folder of model checkpoint.")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")

    # llm 
    args.add_argument('--max_tokens', type=int, default=1024, help="decoding tokens")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")
    args.add_argument('--top_k', type=int, default=-1, help="for sampling")
    args.add_argument('--top_p', type=float, default=1, help="for sampling")
    args.add_argument('--use_beam_search', action="store_true", help="use beam search")
    args.add_argument('--best_of', type=int, default=1, help="for beam search")
    args.add_argument('--n_generate_sample', type=int, default=1, help="number of generated samples")
    args.add_argument('--seed', type=int, default=1234, help="random seed.")
    args.add_argument('--repetition_penalty', type=float, default=1.0, help="repetition penalty")

    args.add_argument('--template', type=str, default="llama3", help="template")

    args.add_argument('--mode', type=str, default="M0", help="M0 or M1")

    args = args.parse_args()
    return args

def input_format(args, d):
    if d.get('input', "") == "":
        if args.lang == "en":
            question = d[args.question_key]
        else:
            if args.with_prefix:
                question = PREFIX.format(language=LANGUAGES[args.lang]) + d[args.question_key]
            else:
                question = d[args.question_key]
    else:
        if args.lang == "en":
            question = d[args.question_key] + "\n" + d['input']
        else:
            if args.with_prefix:
                question = PREFIX.format(language=LANGUAGES[args.lang]) + d[args.question_key] + "\n" + d['input']
            else:
                question = d[args.question_key] + "\n" + d['input']
    if args.verbose:
        print(colored(question, "blue"))
    return question

def cross_implicit_reward_prompt(lang, question_key, template, d):
    if lang == "en":
        question = d[question_key]
    else:
        question = PREFIX.format(language=LANGUAGES[lang]) + d[question_key]

    prompt = SFT_PROMPT[template].format(instruction=question)
       
    if args.verbose:
        print(colored(question, "blue"))
    return prompt

def main(args):
    # init llm
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')

    llm = LLM(
        model=args.checkpoint_dir, 
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=True, 
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        top_k=args.top_k,
        best_of=args.best_of,
        use_beam_search=args.use_beam_search,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_generate_sample,
        repetition_penalty=args.repetition_penalty,
    )


    for lang in args.langs:

        args.lang = lang

        if args.cross_gen:
            args.question_file = os.path.join(args.question_path, f"en.jsonl") if not args.xalpaca_eval else os.path.join(args.question_path, f"en.json")
        else:
            args.question_file = os.path.join(args.question_path, f"{lang}.jsonl") if not args.xalpaca_eval else os.path.join(args.question_path, f"{lang}.json")
        

        args.en_question_file = os.path.join(args.question_path, f"en.jsonl") if not args.xalpaca_eval else os.path.join(args.question_path, f"en.json")
        
        print(colored(f"Start Inference for {args.lang}", "green"))
        print(colored(f"Question File: {args.question_file}", "green"))


        # load question file
        if args.question_file.endswith(".jsonl"):
            data = load_jsonl(args.question_file)

        elif args.question_file.endswith(".json"):
            data = load_json(args.question_file)
        else:
            raise ValueError("Question file should be json or jsonl format.") 
        

        if args.en_question_file.endswith(".jsonl"):
            en_data = load_jsonl(args.en_question_file)

        elif args.en_question_file.endswith(".json"):
            en_data = load_json(args.en_question_file)
        else:
            raise ValueError("Question file should be json or jsonl format.") 

        if "instruction_id" not in en_data[0]:
            en_data = {d["id"]: d for d in en_data}
        else:
            en_data = {d["instruction_id"]: d for d in en_data}

        # write results
        if getattr(args, "save_dir", None) is None and getattr(args, "save_file", None) is None:
            raise ValueError("save_dir or save_file should be provided.")
        elif getattr(args, "save_dir", None) is not None and getattr(args, "save_file", None) is not None:
            raise ValueError("save_dir and save_file cannot be provided at the same time.")
        elif getattr(args, "save_dir", None) is not None:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)
            save_file = os.path.join(args.save_dir, os.path.basename(args.question_file) + f".prediction.with_{args.checkpoint_dir.split('/')[-1]}.temp_{args.temperature}_top-p_{args.top_p}_top-k_{args.top_k}_n-sample_{args.n_generate_sample}_max-tokens_{args.max_tokens}_seed_{args.seed}.to_{args.lang}.jsonl")

            if args.xalpaca_eval:
                save_file = os.path.join(args.save_dir, os.path.basename(args.question_file) + f".prediction.with_{args.checkpoint_dir.split('/')[-1]}.to_{args.lang}.jsonl")

            if args.mode != "M0":
                save_file = os.path.join(args.save_dir, os.path.basename(args.question_file) + f".prediction.with_{args.mode}.temp_{args.temperature}_top-p_{args.top_p}_top-k_{args.top_k}_n-sample_{args.n_generate_sample}_max-tokens_{args.max_tokens}_seed_{args.seed}.to_{args.lang}.jsonl")

        elif getattr(args, "save_file", None) is not None:
            if not os.path.exists(os.path.dirname(args.save_file)):
                os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
            save_file = args.save_file
        
        with open(save_file, "w") as writer:
            for cur_data in tqdm(batch(data, args.num_per_inference), desc="Main Processing"):
                # load question
                questions = []
                for d in cur_data:
                    question = input_format(args, d)
                    questions.append(question)
                
                # inference
                jsonlines = batch_main(args, llm, sampling_params, questions)
                
                
                # write response
                for d in cur_data:
                    question = input_format(args, d)

                    text = d
                    text["prompt"] = jsonlines[question]["prompt"]
                    text["cross_implicit_reward_prompt"] = cross_implicit_reward_prompt(args.lang, args.question_key, args.template, en_data[d["instruction_id"] if "instruction_id" in d else d["id"]])
                    text["response"] = jsonlines[question]["response"]

                    writer.write(json.dumps(text, ensure_ascii=False) + '\n')       
                    writer.flush()
                
if __name__ == '__main__':
    args = parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    main(args)