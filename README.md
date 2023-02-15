# Faithful Baseline Value

## Abstract

Code release for paper ["Can We Faithfully Represent Absence States to Compute Shapley Values on a DNN?"](https://arxiv.org/abs/2105.10719) (accepted by ICLR 2023). Also, the synthetic functions used in the paper is released in `./models/synthetic_functions.xlsx`

> Although many methods have been proposed to estimate attributions of input variables, there still exists a significant theoretical flaw in the masking-based attribution methods, i.e., it is hard to examine whether the masking method faithfully represents the absence of input variables. Specifically, for masking-based attributions, setting an input variable to the baseline value is a typical way of representing the absence of the variable. However, there are no studies investigating how to represent the absence of input variables and verify the faithfulness of baseline values. Therefore, we revisit the feature representation of a deep model in terms of causality, and propose to use causal patterns to examine whether the masking method faithfully removes information encoded in the input variable. More crucially, it is proven that the causality can be explained as the elementary rationale of the Shapley value. Furthermore, we define the optimal baseline value from the perspective of causality, and we propose a method to learn the optimal baseline value. Experimental results have demonstrated the effectiveness of our method.

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n baseline python=3.8
        conda activate baseline
        pip install pip --upgrade
        pip install -r requirements.txt
        conda activate baseline
    ```

## Run Scripts

The command used for learn baseline values for the UCI census income dataset:
```bash
    python main.py --device 1 --seed 1 --dataset census --init zero --lr 0.01 --itr 300 --baseline_bs 1 --sample_num 100 --loss shapley --vfunc log-odds
```

---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[zqs1022@sjtu.edu.cn](zqs1022@sjtu.edu.cn)
