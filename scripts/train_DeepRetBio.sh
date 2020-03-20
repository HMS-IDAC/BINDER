#!/bin/bash
python src/main.py train --epochs=100 --lr=3e-3 --resume --manipulations=manipulations/d.yml \
    $DATA_DIR models/depret.pth --model=DeepRetrieval --bs=256 --loss=hardnet --filetype=.tif --device=cuda:1
 

