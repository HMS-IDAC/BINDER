#!/bin/bash
python src/train.py $TRAIN_PATH $VALID_PATH --learning_rate 1e-2 --resume $CHECKPOINT
