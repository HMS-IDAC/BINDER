import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import random


# Imagenet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


to_256 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])

test_transforms = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

train_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


class TestDataset(Dataset):
    """
    Dataset with image pairs in folders
    """

    def __init__(self, path, filetype=".tif", transforms=test_transforms):
        self.fnames = list(Path(path).glob(f"**/*{filetype}"))
        assert len(self.fnames), f"no files matching **/*{filetype} found"
        self.img_folders = np.array([f.parent.name for f in self.fnames])
        self.classes = np.unique(self.img_folders)
        self.idx = np.arange(len(self.fnames))  # for random selection
        self.transforms = transforms

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):

        anchor_idx = np.random.choice(self.idx[self.img_folders == self.classes[index]])
        same_idx = np.random.choice(
            self.idx[
                (self.idx != anchor_idx) & (self.img_folders == self.classes[index])
            ]
        )
        anchor = Image.open(self.fnames[anchor_idx])
        same = Image.open(self.fnames[same_idx])
        return self.transforms(anchor), self.transforms(same)


class OnlineSyntheticDataset(Dataset):
    """
    Synthetic dataset with online manipulations
    """

    def __init__(self, path, manipulator, filetype=".tif", transforms=train_transforms):
        super(OnlineSyntheticDataset).__init__()
        self.fnames = list(Path(path).glob(f"*{filetype}"))
        assert len(self.fnames), "no files found"
        self.idx = np.arange(len(self))  # for random selection
        self.transforms = transforms
        self.manipulator = manipulator

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        anchor = to_256(Image.open(self.fnames[index]))
        anchor, same = self.manipulator(anchor)

        return self.transforms(anchor), self.transforms(same)


if __name__ == "__main__":
    import argparse
    from manipulate import Manipulator
    from utils import plot_image_pairs, save_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=10, type=int)
    parser.add_argument("data", help="path to data")
    parser.add_argument("manipulations", help="manipulation file")
    parser.add_argument("output", help="output file for images")
    parser.add_argument("--filetype", default=".tif")
    args = parser.parse_args()

    files = list(Path(args.data).glob(f"**/*{args.filetype}"))
    random.shuffle(files)
    manipulator = Manipulator(args.manipulations)

    get_class = lambda f: f.name[1:3]
    classes = set([get_class(f) for f in files])

    pairs = []
    for c in classes:
        # Definitely not the best way to do this but I just need a figure
        class_files = [f for f in files if get_class(f) == c]
        dataset = OnlineSyntheticDataset(class_files, manipulator, triplet=False)
        pairs.append(torch.stack((dataset[0][0], dataset[0][1])))

    save_pairs(torch.stack(pairs), args.output, horizontal=True)
