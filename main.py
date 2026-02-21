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
    
experiment_name = "SCL-HNM_MPQA-P_Default"

papermill.execute_notebook(
    input_path = f'{notebook_name}',
    output_path = f'results/{experiment_name}_{notebook_name}',

    parameters = dict(
        RUNTIME_TYPE = 'CONDA',          # Conda environment
        EXPERIMENT_NAME = experiment_name,
        DATASET_NAME = 'MPQA-P',         # Dataset name
        BASE_MODEL = 'FLAN-T5',          # Base model
        MODEL_NAME = 'google/flan-t5-base', # HuggingFace Model path
        AFL_METHOD = "Rep(En)-ClUn(En)", 

        SAMPLING_ITERATIONS = 10,        # Iterations / epochs
        NUMBER_OF_SAMPLES_PER_ITERATION = 10, # samples per epoch
   
        REPEAT_SAMPLING = 5,             # Seeds / Number of experiments

        STORE_RESULTS = ['eval_polarity_accuracy', 'eval_polarity_weighted_f1'], 
        RESET_MODEL_AFTER_EACH_ITERATION = True, 
        INITIAL_LEARNING_RATE = 1e-4,    
        LEARNING_RATE_DECAY = 1,

        USE_CONTRASTIVE_LOSS = True,     # Contrastive learning switch
        CONTRASTIVE_LOSS_ALPHA = 0.1,    # Contrastive loss weight
        SIMILARITY_METRIC = 'cosine',    # Similarity metric: 'cosine' or 'euclidean'
        TEMPERATURE = 0.05,              # Temperature tau
        
        USE_HARD_NEGATIVE_MINING = True, # Hard negative mining switch
        HARD_NEGATIVE_MINING_M = 5,      # Number of hard negatives M

        # MPQA Polarity
        mpqa_p_train_link = './data/MPQA_dataset/mpqa_polarity_train.csv',
        mpqa_p_val_link   = './data/MPQA_dataset/mpqa_polarity_val.csv',
        mpqa_p_test_link  = './data/MPQA_dataset/mpqa_polarity_test.csv',
    )
)
