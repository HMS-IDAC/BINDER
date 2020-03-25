import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from manipulate import Manipulator
from models import Retrieval
from loss import hardest_loss
from dataset import TestDataset, OnlineSyntheticDataset


class RetrievalModule(pl.LightningModule):
    def __init__(self, hparams):
        super(RetrievalModule, self).__init__()
        self.hparams = hparams
        self.model = Retrieval(base=hparams.base, chunks=hparams.chunks)
        self.manipulator = Manipulator(self.hparams.manipulations)

    def forward(self, a, b):
        return self.model(a), self.model(b)

    def training_step(self, batch, batch_idx):
        anchor, same = batch
        anchor_emb, same_emb = self.forward(anchor, same)
        loss = hardest_loss(anchor_emb, same_emb)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        anchor, same = batch
        anchor_emb, same_emb = self.forward(anchor, same)
        loss = hardest_loss(anchor_emb, same_emb)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.fc.parameters()},
                {"params": self.model.base.parameters()},
                {
                    "params": self.model.gem.parameters(),
                    "lr": self.hparams.learning_rate * 10,
                },
            ],
            lr=self.hparams.learning_rate,
        )

    def train_dataloader(self):
        train_set = OnlineSyntheticDataset(
            self.hparams.train_path, self.manipulator, filetype=self.hparams.filetype
        )
        return DataLoader(
            train_set, num_workers=4, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        if self.hparams.synth_valid:
            valid_set = OnlineSyntheticDataset(
                self.hparams.valid_path,
                self.manipulator,
                filetype=self.hparams.filetype,
            )
        else:
            valid_set = TestDataset(
                self.hparams.valid_path, filetype=self.hparams.filetype
            )
        return DataLoader(valid_set, num_workers=4, batch_size=self.hparams.batch_size)


def main(args):
    if args.resume:
        module = RetrievalModule.load_from_checkpoint(args.resume)
    else:
        module = RetrievalModule(args)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus)
    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--base", default="resnet50")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--manipulations", default="manipulations.yml")
    parser.add_argument("--synth_valid", action="store_true", default=False)
    parser.add_argument("--filetype", default=".tif")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--resume", type=str, default=False)

    parser.add_argument("train_path", help="path to training imgs")
    parser.add_argument("valid_path", help="path to validation imgs")

    args = parser.parse_args()
    main(args)
