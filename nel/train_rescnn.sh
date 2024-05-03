#!/bin/bash

# Set your own path for virtual environment
source /jet/home/ghong1/miniconda3/bin/activate rescnn
echo "Activated rescnn"

# Step 1: Train ResCNN EL model
# This step includes fine-tuning EL model, 
# as well as generating prediction file for gold mentions

python cg_trainer.py \
    --cg_config lightweight_cnn_text_with_attention_pooling_biolinkbert \
    --dataset bc8biored-disease-aio > ./logs/bc8biored-disease-aio-biolinkbert-attnpooling.txt \
    --seed 5

depth_c=3
python cg_trainer.py \
    --cg_config lightweight_cnn_text_biolinkbert \
    --dataset bc8biored-chemical-aio > ./logs/bc8biored-chemical-aio-biolinkbert-maxpooling.txt \
    --cnn_text_depth $depth_c

# Step 2: Infer IDs of predicted mentions in the dev set 
# with fine-tuned EL model

python cg_trainer.py \
    --cg_config lightweight_cnn_text_with_attention_pooling_biolinkbert \
    --dataset bc8biored-disease-aio > ./logs/bc8biored-disease-aio-biolinkbert-attnpooling-pred-dev.txt \
    --do_predict_dev

depth_c=3
python cg_trainer.py \
    --cg_config lightweight_cnn_text_biolinkbert \
    --dataset bc8biored-chemical-aio > ./logs/bc8biored-chemical-aio-biolinkbert-maxpooling-pred-dev.txt \
    --cnn_text_depth $depth_c \
    --do_predict_dev
