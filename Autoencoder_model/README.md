## Autoencoder Model 

Downloads |     |
---       | --- |
[weights](https://www.dropbox.com/s/cxtytmpsc749rkt/weights.zip?dl=0)| Folder with weights 
Contents of folder
..* Autoencoder_base_pretrained_COCO.hdf5| Autoencoder base network pretrained on COCO
..* Autoencoder_top_pretrained_COCO.hdf5| Autoencoder model top fully-connected layers pretrained on COCO 
..* Autoencoder_top_finetuned_BIO.hdf5| Autoencoder model top fully-connected layers finetuned on BINDER 

### Test Pre-trained Models

```bash
python src/main_autoencode.py \
       --t_path='path/to/binder/test/' \
       --d_name=BINDER \
       --input_weights_path='./weights/Autoencoder_top_finetuned_BIO.hdf5'
       
       Or
       
./scripts/test_model.sh           
```

Use `--d_name=BINDER` to test on BINDER dataset, `--d_name=MFND` to test on MFND images, `--input_weights_path='./weights/Autoencoder_top_fnetuned_BIO.hdf5'` to test model fine-tuned on BINDER & `--input_weights_path='./weights/Autoencoder_top_pretrained_COCO.hdf5'` to test with model pre-trained on `COCO`  

### Train Models

#### Pre-train

```bash
./scripts/train_Autoencoder_base_COCO.sh  # pre-train Autoencoder base network
./scripts/train_Autoencoder_top_COCO.sh   # pre-train Autoencodermodel top layers
```

#### Fine-tune

```bash
./scripts/finetune_Autoencoder_top_BIO.sh  # fine-tune Autoencoder top layers, base network is pretrained on COCO 
```

