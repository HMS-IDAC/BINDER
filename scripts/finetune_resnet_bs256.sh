#!/bin/bash
cp models/pretrain_resnet_bs256.pth models/finetune_resnet_bs256.pth;

python src/main.py train \
    --resume \
    --bs=256 \
    --loss=hardest \
    --filetype=.tif \
    --device=cuda:0 \
    --epochs=100 \
    --lr=1e-1 \
    --train_path=$TRAIN_PATH --valid_path=$VALID_PATH \
    --manipulations=manipulations.yml \
    --weights_path=models/finetune_resnet_bs256.pth \
    --base=resnet50 \
    --chunks=8
