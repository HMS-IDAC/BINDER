import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from sklearn.metrics import roc_auc_score, roc_curve

from utils import save_farthest_duplicates, save_closest_diff, hard_mine
from loss import hardest_loss
from models import BioMetric
from dataset import TestDataset, OnlineSyntheticDataset, loaders
from manipulate import Manipulator

loss_fns = {"triplet": torch.nn.TripletMarginLoss(), "hardest": hardest_loss}


class Trainer:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.model = BioMetric(chunks=args.chunks, base=args.base).to(self.device)
        self.criterion = loss_fns[args.loss]
        self.dist_fn = F.pairwise_distance
        self.bs = args.bs
        self.weights_path = args.weights_path

        self.triplet = False if args.loss == "hardest" else True

    def train(self, args):

        if args.resume:
            self.model.load_state_dict(torch.load(self.weights_path))

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.fc.parameters()},
                {"params": self.model.base.parameters()},
                {"params": self.model.gem.parameters(), "lr": args.lr * 10},
            ],
            lr=args.lr,
        )

        manipulator = Manipulator(args.manipulations)

        train_loader, valid_loader = loaders(
            args.train_path,
            args.valid_path,
            manipulator,
            self.bs,
            filetype=args.filetype,
            triplet=self.triplet,
            synthetic_valid=args.synth_valid,
        )

        # Tensorboard
        self.writer = SummaryWriter()

        best_loss = float("inf")
        print("train loss\tvalid loss")
        for epoch in range(args.epochs):
            self.model.train()
            train_loss = self._train_step(train_loader)
            self.model.eval()
            with torch.no_grad():
                valid_loss = self._test_step(valid_loader)[0]
            self.writer.add_scalars(
                "loss", {"train": train_loss, "valid": valid_loss}, epoch
            )
            self.writer.flush()
            print(f"{train_loss:.4f}\t{valid_loss:.4f}")

            if valid_loss < best_loss:
                print("saving")
                torch.save(self.model.state_dict(), self.weights_path)
                best_loss = valid_loss

    def test(self, args):
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.eval()
        dataloader = DataLoader(
            TestDataset(args.test_path, args.filetype, triplet=self.triplet),
            batch_size=self.bs,
        )
        if self.triplet:
            labels, distances = [], []
        else:
            anchor_emb, same_emb = [], []
        losses, pairs = [], []

        with torch.no_grad():
            for _ in tqdm(range(args.n)):
                if self.triplet:
                    loss, label, dist, pair = self._test_step(
                        dataloader, save_pairs=True
                    )
                    labels.append(label)
                    distances.append(dist)
                else:
                    loss, anchor, same, pair = self._test_step(
                        dataloader, save_pairs=True
                    )
                    anchor_emb.append(anchor)
                    same_emb.append(same)
                losses.append(loss)
                pairs.append(pair)

        if self.triplet:
            labels = torch.cat(labels).numpy()
            distances = torch.cat(distances).numpy()
            pairs = torch.cat(pairs).numpy()
        else:
            distances, labels, pairs, dist_mat = hard_mine(
                torch.cat(anchor_emb), torch.cat(same_emb), torch.cat(pairs)
            )
            distances, labels = distances.numpy(), labels.numpy()

        print("AUC", roc_auc_score(labels, 1 / (distances + 1e-8)))

        if args.save_roc is not None:
            fpr, tpr, _ = roc_curve(labels, 1 / (distances + 1e-8))
            roc = np.stack([fpr, tpr])
            np.save(args.save_roc, roc)

        if args.save_farthest_dup is not None:
            save_farthest_duplicates(labels, distances, pairs, args.save_farthest_dup)

        if args.save_closest_diff is not None:
            save_closest_diff(labels, distances, pairs, args.save_closest_diff)

    def _forward(self, data):
        if self.triplet:
            anchor, same, diff = data
            anchor, same, diff = (
                anchor.to(self.device),
                same.to(self.device),
                diff.to(self.device),
            )
            return self.model(anchor), self.model(same), self.model(diff)
        anchor, same = data
        anchor, same = anchor.to(self.device), same.to(self.device)
        return self.model(anchor), self.model(same)

    def _train_step(self, dataloader):
        """Single training step through data"""
        total_loss = 0.0
        pbar = tqdm(dataloader, leave=False)
        for data in pbar:
            self.optimizer.zero_grad()
            outs = self._forward(data)
            loss = self.criterion(*outs)
            total_loss += loss.item() * outs[0].size(0)
            loss.backward()
            self.optimizer.step()

        total_loss /= len(dataloader.dataset)
        return total_loss

    def _test_step(self, dataloader, save_pairs=False):
        """Single test/valid step through data, optionally save image pairs"""
        total_loss = 0.0
        pbar = tqdm(dataloader, leave=False)
        if self.triplet:
            labels, distances, pairs, img, emb = [], [], [], [], []
        else:
            anchor_emb, same_emb, pairs = [], [], []
        for data in pbar:
            outs = self._forward(data)
            total_loss += self.criterion(*outs).item() * outs[0].size(0)

            if self.triplet:
                # if triplets are provided, store distances + labels for evaluation metrics
                same_dist = self.dist_fn(outs[0], outs[1])
                diff_dist = self.dist_fn(outs[0], outs[2])
                same_label = torch.ones_like(same_dist)
                diff_label = torch.zeros_like(diff_dist)
                distances.append(torch.cat((same_dist, diff_dist)))
                labels.append(torch.cat((same_label, diff_label)))

                if save_pairs:
                    img.append(data[0])
                    emb.append(outs[0])
                    same_imgs = torch.stack([data[0], data[1]], dim=1)
                    diff_imgs = torch.stack([data[0], data[2]], dim=1)
                    pairs.append(torch.cat((same_imgs, diff_imgs)))
            else:
                if save_pairs:
                    anchor_emb.append(outs[0])
                    same_emb.append(outs[1])
                    pairs.append(torch.stack([data[0], data[1]], dim=1))

        total_loss /= len(dataloader.dataset)
        if self.triplet:
            labels = torch.cat(labels).cpu()
            distances = torch.cat(distances).cpu()
            if save_pairs:
                pairs = torch.cat(pairs)
                # NOTE: uncomment to save embedings for TSNE
                # TODO: should make this an option
                # self.writer.add_embedding(torch.cat(emb), label_img=torch.cat(img))
                return total_loss, labels, distances, pairs
            return total_loss, labels, distances
        if save_pairs:
            anchor_emb = torch.cat(anchor_emb).cpu()
            same_emb = torch.cat(same_emb).cpu()
            pairs = torch.cat(pairs)
        return total_loss, anchor_emb, same_emb, pairs


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("action", choices=["train", "test"])

    parser.add_argument("--base", help="base architecture for model", default="resnet50")
    parser.add_argument("--loss", choices=loss_fns.keys(), default="triplet")
    parser.add_argument("--chunks", type=int, default=0, help="number of chunks for gradient checkpointing")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--filetype", default=".tif")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights_path", help="path to model weights")

    # train args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--manipulations", default="manipulations/a.yml")
    parser.add_argument("--train_path", help="path to train data")
    parser.add_argument("--valid_path", help="path to valid data")
    parser.add_argument("--synth_valid", action="store_true", default=False, help="use synthetic manipulations for validation")

    # test args
    parser.add_argument("--test_path", help="path to test data")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save_roc", help="file to save roc")
    parser.add_argument("--save_farthest_dup", help="file to save farthest duplicates")
    parser.add_argument("--save_closest_diff", help="file to save closest different")

    # fmt: on

    # call appropriate function
    args = parser.parse_args()
    trainer = Trainer(args)

    if args.action == "train":
        trainer.train(args)
    elif args.action == "test":
        trainer.test(args)
