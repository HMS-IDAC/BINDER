#!/bin/bash
mkdir -p models
python src/main.py train \
    --bs=256 \
    --loss=hardest \
    --filetype=.jpg \
    --device=cuda:0 \
    --epochs=100 \
    --lr=1e-1 \
    --train_path=$TRAIN_PATH --valid_path=$VALID_PATH \
    --synth_valid \
    --manipulations=manipulations.yml \
    --weights_path=models/pretrain_resnet_bs256.pth \
    --base=resnet50 \
    --chunks=8
