#!/bin/bash
mkdir -p models
python src/main.py train \
    --bs=256 \
    --loss=hardest \
    --filetype=.tif \
    --device=cuda:0 \
    --epochs=200 \
    --lr=3e-2 \
    --train_path=$TRAIN_PATH --valid_path=$VALID_PATH \
    --manipulations=manipulations.yml \
    --weights_path=models/train_resnet_bs256.pth \
    --base=resnet50 \
    --chunks=8
