#!/bin/bash
mkdir -p models
python src/main.py train \
    --bs=256 \
    --loss=hardest \
    --filetype=.jpg \
    --device=cuda:1 \
    --epochs=100 \
    --lr=1e-1 \
    --train_path=$TRAIN_PATH --valid_path=$VALID_PATH \
    --synth_valid \
    --manipulations=manipulations.yml \
    --weights_path=models/pretrain_vgg_bs256.pth \
    --base=vgg19 \
    --chunks=8
