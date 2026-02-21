#!/usr/bin/python

import os
import cgi
import json
import logging
import papermill
from pathlib import Path
from urllib.request import urlopen

os.environ["TOKENIZERS_PARALLELISM"] = "false"

Path("notebooks/").mkdir(parents=True, exist_ok=True)
Path("results/").mkdir(parents=True, exist_ok=True)

notebook_name = "afl.ipynb"
    
experiment_name = "SCL-HNM_AMZN-EN_Default"

papermill.execute_notebook(
    input_path = f'{notebook_name}',
    output_path = f'results/{experiment_name}_{notebook_name}',

    parameters = dict(
        RUNTIME_TYPE = 'CONDA',          # Conda environment
        EXPERIMENT_NAME = experiment_name,
        DATASET_NAME = 'AMZN-EN',        # Dataset name
        BASE_MODEL = 'FLAN-T5',          # Base model
        MODEL_NAME = 'google/flan-t5-base', # HuggingFace Model path
        AFL_METHOD = "Rep(En)-ClUn(En)", 

        SAMPLING_ITERATIONS = 10,        # Iterations / epochs
        NUMBER_OF_SAMPLES_PER_ITERATION = 10, # samples per epoch
   
        REPEAT_SAMPLING = 5,             # Seeds / Number of experiments

        STORE_RESULTS = ['eval_main_metric'], 
        RESET_MODEL_AFTER_EACH_ITERATION = True, 
        INITIAL_LEARNING_RATE = 1e-4,    
        LEARNING_RATE_DECAY = 1,

        USE_CONTRASTIVE_LOSS = True,     # 对比学习总开关
        CONTRASTIVE_LOSS_ALPHA = 0.1,   #  lambda
        SIMILARITY_METRIC = 'cosine',    # 相似度度量: 'cosine' 或 'euclidean'
        TEMPERATURE = 0.05,              # 温度系数 tau
        
        USE_HARD_NEGATIVE_MINING = True, # 硬样本挖掘开关
        HARD_NEGATIVE_MINING_M = 5,      # 硬负样本数量 M

        # # MPQA Type
        # mpqa_t_train_link = './data/MPQA_dataset/mpqa_type_train.csv',
        # mpqa_t_val_link   = './data/MPQA_dataset/mpqa_type_val.csv',
        # mpqa_t_test_link  = './data/MPQA_dataset/mpqa_type_test.csv',

        # MPQA Polarity
        STORE_RESULTS = ['eval_polarity_accuracy', 'eval_polarity_weighted_f1'],
        mpqa_p_train_link = './data/MPQA_dataset/mpqa_polarity_train.csv',
        mpqa_p_val_link   = './data/MPQA_dataset/mpqa_polarity_val.csv',
        mpqa_p_test_link  = './data/MPQA_dataset/mpqa_polarity_test.csv',

        # # MPQA Intensity
        # mpqa_i_train_link = './data/MPQA_dataset/mpqa_intensity_train.csv',
        # mpqa_i_val_link   = './data/MPQA_dataset/mpqa_intensity_val.csv',
        # mpqa_i_test_link  = './data/MPQA_dataset/mpqa_intensity_test.csv',

        # # AG News
        # agnews_train_link = './data/AG_news_dataset/agnews_train.csv',
        # agnews_test_link  = './data/AG_news_dataset/agnews_test.csv',

        # # Amazon Reviews
        # amzn_train_link = './data/Amazon_dataset/amazon_train.csv',
        # amzn_val_link   = './data/Amazon_dataset/amazon_val.csv',
        # amzn_test_link  = './data/Amazon_dataset/amazon_test.csv',
    )
)