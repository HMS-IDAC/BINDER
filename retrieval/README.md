
## Test Pre-trained Models

Download models:
```bash
./scripts/download_weights.sh
```

Test model:
```bash
python test.py path/to/binder/test models/model.ckpt
```

Additionally you can use the `--save_closest_diff`, `--save_farthest_dup`, and
`--save_roc` flags, to save the closest different images, farthest duplicate
images, and ROC curve data respectively.


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
# path to pre-trained checkpoint
export CHECKPOINT="lightning_logs/version_N/checkpoints/epoch=E.ckpt"
./scripts/finetune.sh
```

