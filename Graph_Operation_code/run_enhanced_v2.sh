#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_type devign \
    --dataset devign_enhanced_v2 \
    --input_dir ../devign_dataset/devign_cpg_enhanced_v2 \
    --feature_size 768 \
    --graph_embed_size 768 \
    --num_steps 6 \
    --batch_size 32 \
    --log_dir devign_enhanced_v2.log
