# BINDER

![train](figures/train.png)

**On Identification and Retrieval of Near-Duplicate Biological Images: a New Dataset and Baseline**<br>
T.E. Koker, S.S. Chintapalli, S. Wang, B.A. Talbot, D. Wainstock, M. Cicconet, and M. Walsh

Paper:<br>
Dataset: idac.hms.harvard.edu/binder<br>
Website:

Abstract: *Manipulation and re-use of images in scientific publications is a
growing issue, not only for biomedical publishers, but also for the research
community in general. In this work we introduce BINDER -- Bio-Image
Near-Duplicate Examples Repository, a novel dataset to help researchers develop,
train, and test models to detect same-source biomedical images. BINDER contains
7,490 unique image patches -- the training set -- as well as 1,821 patches --
split in validation and test sets -- with accompanying manipulations obtained by
means of transformations including rotation,  translation,  scale,  perspective
transform, contrast adjustment and/or compression artifacts. In addition to the
dataset, we show how existing image retrieval and metric learning models can be
applied to achieve high-accuracy inference results, creating a baseline for
future work. The dataset is available at idac.hms.harvard.edu/binder, and
the source code for models and manipulations will be available upon
publication.*


Downloads |     |
---       | --- |
[finetune_resnet_bs256.pth](https://www.dropbox.com/s/56s9p5felbjptj5/finetune_resnet_bs256.pth?dl=0)| Fine-tuned ResNet50 weights
[finetune_vgg_bs256.pth](https://www.dropbox.com/s/iv22fdxxt8damha/finetune_vgg_bs256.pth?dl=0)| Fine-tuned VGG19 weights
[pretrain_resnet_bs256.pth](https://www.dropbox.com/s/5jgv8ixefvp1gtw/pretrain_resnet_bs256.pth?dl=0)| Pre-trained ResNet50 weights
[pretrain_vgg_bs256.pth](https://www.dropbox.com/s/fs1co4b60tn5cqw/pretrain_vgg_bs256.pth?dl=0)| Pre-trained VGG19 weights

## Test Pre-trained Models

```bash
python src/main.py test \
    --base="resnet50" \  # or "vgg19"
    --loss="hardest" \   # or "triplet" for random negatives
    --chunks=8 \         # number of chunks to use for gradient checkpoints
    --bs=256 \           # batch size
    --filetype=".tif" \
    --device="cuda:0" \
    --weights_path=pretrain_resnet_bs256.pth \  # path to model weights
    --test_path=/path/to/binder/test/
```

## Using Manipulator

See `manipulations.yml` for configuration used to generate dataset

```python
from PIL import Image
from src.manipulate import Manipulator

manipulator = Manipulator("manipulations.yaml")

img = Image.open("path/to/image")

anchor, same = manipulator(img)  # creates duplicate pair
```
## Train Models

### Pre-train

```bash
export TRAIN_PATH=/path/to/coco/train
export VALID_PATH=/path/to/coco/valid
./scripts/pretrain_resnet_bs256.sh  # pre-train ResNet
./scripts/pretrain_vgg_bs256.sh     # pre-train VGG
```

### Fine-tune

```bash
export TRAIN_PATH=/path/to/binder/train
export VALID_PATH=/path/to/binder/valid
./scripts/finetune_resnet_bs256.sh
./scripts/finetune_vgg_bs256.sh
```

