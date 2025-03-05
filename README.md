# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment

<div align="center">
<br>
<a>Wen Yang</a><sup><span>1,2</span></sup>, 
<a href="https://scholar.google.com/citations?user=Ci4l4yQAAAAJ&hl=zh-CN">Junhong Wu</a><sup><span>1,2</span></sup>,
<a href="https://scholar.google.com/citations?user=FgrrqlAAAAAJ&hl=zh-CN">Chen Wang</a><sup><span>1,2</span></sup>,
<a href="https://scholar.google.com/citations?user=l8lvKOQAAAAJ&hl=zh-CN">Chengqing Zong</a><sup><span>1,2</span></sup>,
<a href="https://scholar.google.com/citations?user=93zngeYAAAAJ&hl=zh-CN">Jiajun Zhang</a><sup><span>1,2,3,4🌟</span></sup>,
<br>
    
🌟 Corresponding author

<sup>1</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
<sup>2</sup> Institute of Automation, Chinese Academy of Sciences<br>
<sup>3</sup> Wuhan AI Research
<sup>4</sup> Shanghai Artificial Intelligence Laboratory, Shanghai, China<br>
    
![Multilingual-Preference-Optimization](https://img.shields.io/badge/Task-Multilingual--Preference--Optimization-red) <a href='https://arxiv.org/pdf/2410.08964'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
</div>

## Overview
We introduce **Language Imbalance Driven Rewarding**, a novel approach that leverages the inherent capability imbalance across different languages in large language models (LLMs) as a reward signal for iterative self-improvement. By applying iterative DPO training, our approach _not only enhances the performance of non-dominant languages but also improves outcomes in dominant languages._ 

Our goal with this approach is to contribute a new perspective to the multilingual LLM community by challenging the assumption that language imbalance is solely a challenge to be mitigated. We hope this approach will inspire further exploration into _multilingual self-improvement_ in LLMs, broadening the horizon for more balanced and capable language models.

## 🔥 Update

- [06/03/2025]🔥We release the [code](https://github.com/ZNLP/Implicit-Cross-Lingual-Rewarding) for Implicit Cross-Lingual Rewarding!
- [05/03/2025]🔥Implicit Cross-Lingual Rewarding is coming! We release the [paper](https://arxiv.org/pdf/2405.15232)!

## 👀 Contents

- [Setup](#Setup)
- [Preparation](#Preparation)
- [Train](#Train)
- [Evaluation](#Evaluation)
- [Experiments](#Experiments)
- [Citation](#citation)


## 📷 Setup

Please follow the instructions below to install the required packages.


1. Clone this repository

```bash
https://github.com/ZNLP/Implicit-Cross-Lingual-Rewarding.git
```

2. Install Package

```bash
conda create -n ICR python=3.10 -y
conda activate ICR
cd Implicit-Cross-Lingual-Rewarding
pip install -r requirements.txt
```

## 💡 Preparation


```bash
bash ./scripts/run_pipeline.sh
```

## 📈 Train

Our training is mostly performed on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) code base. Please refer to that repo for more details.

## 📈 Inference_on_X-AlpacaEval

```bash
bash scripts/batch_inference_for_xalpacaeval.sh
```

## 👀 Experiments

We provide some results in this section. More detailed results can be found in our paper.

### General Instruction Following

+ X-alpacaEval Leaderboard
<div align=center>
<img width="90%" src="assets/x-alpacaeval.png"/>
</div>

<div align='center'>
<details>
<summary>Click to expand more examples</summary>
<p align="center">
    <img src="assets/mt_bench.png" width="60%" height="60%">
    <p align="center">The Multilingual MT-Bench Benchmark</p>
    <img src="assets/multilingual_NLP_tasks.png" width="60%" height="60%">
    <p align="center">The Multilingual NLP Benchmarks</p>
</p>
</details>
</div>

### More Analysis
+ Different Implicit Rewards
<div align=center>
<img width="80%" src="assets/different_reward_models.png"/>
</div>


<div align='center'>
<details>
<summary>Click to expand more examples</summary>
<p align="center">
    <img src="assets/low_resource_languages.png" width="60%" height="60%">
    <p align="center">Generalization to Lower-resource Languages</p>
    <img src="assets/different_prompt_numbers.png" width="60%" height="60%">
    <p align="center">Scaling the Number of Training Prompts</p>
</p>
</details>
</div>

## Citation

If you find this repo useful for your research, please consider citing the paper

```
TBD
```

## Acknowledgement

We would like to thank the following repos for their great work:
- This work utilizes the great work from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Vllm](https://github.com/vllm-project/vllm), [transformers](https://github.com/huggingface/transformers), [LLaMA](https://github.com/facebookresearch/llama), [SimPO](https://github.com/princeton-nlp/SimPO)
## License

This project is released under the [Apache 2.0 license](https://github.com/RainBowLuoCS/DEEM/blob/main/LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.
