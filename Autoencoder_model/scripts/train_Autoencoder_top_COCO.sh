#!/bin/bash

python src/main_autoencode.py --t_path='/home/sc648/Documents/keras/COCO_data/train2017/' --v_path='/home/sc648/Documents/keras/COCO_data/val2017/' --output_weights_path='./weights/Autoencoder_top_pretrained_COCO.hdf5' --model='Autoencoder_top' --dataset='COCO' 
