#!/bin/bash 
python src/train.py $TRAIN_PATH $VALID_PATH --synth_valid --base=resnet50 --filetype ".jpg"

