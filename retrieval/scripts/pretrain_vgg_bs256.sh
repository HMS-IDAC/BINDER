#!/bin/bash 
python src/train.py $TRAIN_PATH $VALID_PATH --synth_valid --base=vgg19
