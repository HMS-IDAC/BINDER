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


def train_loaders(
    files, manipulator, batch_size, triplet, p_hard=None, get_class=None, split=0.9
):
    random.shuffle(files)
    train_size = int(len(files) * 0.9)
    train_files, valid_files = files[:train_size], files[train_size:]
    loaders = []
    for i, fs in enumerate([train_files, valid_files]):
        loader = DataLoader(
            OnlineSyntheticDataset(
                fs, manipulator, p_hard=p_hard, triplet=triplet, get_class=get_class
            ),
            shuffle=(i == 0),  # shuffle training set only
            batch_size=batch_size,
            num_workers=4,
        )
        loaders.append(loader)
    return loaders


def loaders(
    train_path,
    valid_path,
    manipulator,
    batch_size,
    filetype=".tif",
    triplet=True,
    synthetic_valid=True,
):
    train_files = list(Path(train_path).glob(f"**/*{filetype}"))
    train_set = OnlineSyntheticDataset(train_files, manipulator, triplet=triplet)
    if synthetic_valid:
        valid_files = list(Path(valid_path).glob(f"**/*{filetype}"))
        valid_set = OnlineSyntheticDataset(valid_files, manipulator, triplet=triplet)
    else:
        valid_set = TestDataset(Path(valid_path), filetype=filetype, triplet=triplet)

    ls = []
    for i, ds in enumerate([train_set, valid_set]):
        loader = DataLoader(
            ds,
            shuffle=(i == 0),  # shuffle training set only
            batch_size=batch_size,
            num_workers=4,
        )
        ls.append(loader)

    return ls


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
    def __init__(
        self, image_dir, filetype=".tif", triplet=True, transforms=test_transforms
    ):
        self.fnames = list(Path(image_dir).glob(f"**/*{filetype}"))
        assert len(self.fnames), f"no files matching **/*{filetype} found"
        self.img_folders = np.array([f.parent.name for f in self.fnames])
        self.classes = np.unique(self.img_folders)
        self.triplet = triplet
        self.idx = np.arange(len(self.fnames))  # for random selection
        self.transforms = transforms

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):

        if self.triplet:
            anchor_idx = np.random.choice(
                self.idx[self.img_folders == self.classes[index]]
            )
            same_idx = np.random.choice(
                self.idx[
                    (self.idx != anchor_idx) & (self.img_folders == self.classes[index])
                ]
            )
            diff_idx = np.random.choice(
                self.idx[self.img_folders != self.classes[index]]
            )

            diff = Image.open(self.fnames[diff_idx])
            anchor = Image.open(self.fnames[anchor_idx])
            same = Image.open(self.fnames[same_idx])

            return (
                self.transforms(anchor),
                self.transforms(same),
                self.transforms(diff),
            )

        anchor_idx = np.random.choice(self.idx[self.img_folders == self.classes[index]])
        same_idx = np.random.choice(
            self.idx[
                (self.idx != anchor_idx) & (self.img_folders == self.classes[index])
            ]
        )
        anchor = Image.open(self.fnames[anchor_idx])
        same = Image.open(self.fnames[same_idx])
        return (self.transforms(anchor), self.transforms(same))


class OnlineSyntheticDataset(Dataset):
    """
    Synthetic dataset with online manipulations

    example of get_class: lambda f: f.name[1:3]
    """

    def __init__(
        self,
        fnames,
        manipulator,
        transforms=train_transforms,
        p_hard=None,
        triplet=True,
        get_class=None,
    ):
        super(OnlineSyntheticDataset).__init__()
        self.fnames = fnames
        assert len(self.fnames), "no files found"
        if triplet:
            if p_hard:
                # classes are only important for "hard" cases
                assert (
                    get_class is not None
                ), "must provide get_class function for hard cases"
                self.classes = np.array([get_class(f) for f in self.fnames])
        self.idx = np.arange(len(self))  # for random selection
        self.transforms = transforms
        self.manipulator = manipulator
        self.triplet = triplet
        self.p_hard = p_hard

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        anchor = to_256(Image.open(self.fnames[index]))
        anchor, same = self.manipulator(anchor)

        if not self.triplet:
            return (self.transforms(anchor), self.transforms(same))

        if self.p_hard is not None:
            if np.random.random() < self.p_hard:
                # select hard image (same class)
                diff_idx = np.random.choice(
                    self.idx[
                        (self.idx != index) & (self.classes == self.classes[index])
                    ]
                )
            else:
                # select easy image (different class)
                diff_idx = np.random.choice(
                    self.idx[
                        (self.idx != index) & (self.classes != self.classes[index])
                    ]
                )
        else:
            # select any image thats not the same
            diff_idx = np.random.choice(self.idx[(self.idx != index)])

        diff = Image.open(self.fnames[diff_idx])
        _, diff = self.manipulator(diff)  # only use manipulated diff image
        return (self.transforms(anchor), self.transforms(same), self.transforms(diff))


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
