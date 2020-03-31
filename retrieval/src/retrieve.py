import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from pathlib import Path
from sklearn.neighbors import KDTree
import pandas as pd

from dataset import MEAN, STD, test_transforms
from train import RetrievalModule


class RetrievalDataset(Dataset):
    def __init__(self, path, filetype="*.tif", transforms=test_transforms):
        self.fnames = list(Path(path).glob(f"**/*{filetype}"))
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img = Image.open(self.fnames[index])
        return str(self.fnames[index]), self.transforms(img)


def step(model, batch):
    fname, img = batch
    emb = model.model(img)
    return {"fname": fname, "emb": emb}


@torch.no_grad()
def get_embeddings(model, loader):
    step_out = [step(model, batch) for batch in loader]
    embs = torch.cat([out["emb"] for out in step_out]).cpu().numpy()
    fnames = sum([list(out["fname"]) for out in step_out], [])
    return fnames, embs


def main(args):
    model = RetrievalModule.load_from_checkpoint(args.checkpoint)
    dataset = RetrievalDataset(args.path, filetype=args.filetype)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=model.hparams.batch_size)
    fnames, embs = get_embeddings(model, dataloader)

    # KDTree with embeddings
    tree = KDTree(embs)
    dist, ind = tree.query(embs, k=args.k, sort_results=True)

    data = {}
    data["img"] = fnames
    for k in range(args.k):
        data[f"match_{k}"] = [fnames[i] for i in ind[:, k]]

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filetype", default=".tif")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--o", default="out.csv")
    parser.add_argument("path")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args)
