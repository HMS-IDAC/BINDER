#!/bin/bash

python src/main_autoencode.py --t_path='/home/sc648/Documents/keras/Cell_newdataset/ImageForensicsDataset2020Jan30/FinalDatasetCleanJpeg/train/' --v_path='/home/sc648/Documents/keras/Cell_newdataset/ImageForensicsDataset2020Jan30/FinalDatasetCleanJpeg/valid/' --output_weights_path='./weights/Autoencoder_top_finetuned_BIO.hdf5' --batch_size=16 --model='Autoencoder_top' --dataset='BINDER' --input_weights_path='./weights/Autoencoder_top_pretrained_COCO.hdf5'

