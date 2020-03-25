import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

plt.rcParams["font.family"] = "Times New Roman"

MODELS = {
    "pretrain_vgg": ("Pretrained VGG", "#7570b3"),
    "finetune_vgg": ("Finetuned VGG", "#7570b3"),
    "pretrain_resnet": ("Pretrained ResNet", "#1b9e77"),
    "finetune_resnet": ("Finetuned ResNet", "#1b9e77"),
    "pretrain_autoencoder": ("Pretrained Autoencoder", "#d95f02"),
    "finetune_autoencoder": ("Finetuned Autoencoder", "#d95f02"),
}

DATASETS = {"bio": "BINDER Test", "real": "PUBPEER", "mfnd": "MFND IND"}


def plot_roc(args):
    files = list(Path(args.path).iterdir())
    files.sort()
    fig = plt.figure(figsize=(16, 4))
    num_figs = len(DATASETS)
    for i, (key, name) in enumerate(DATASETS.items()):
        ax = fig.add_subplot(1, num_figs, i + 1)
        for file in files:
            if key in file.name:
                fpr, tpr = np.load(file)
                label = file.stem.split("_", 1)[1]
                if label in MODELS:
                    if not "pretrain" in label:
                        ax.plot(
                            fpr,
                            tpr,
                            label=MODELS[label][0],
                            color=MODELS[label][1],
                            linestyle=":",
                        )
                    else:
                        ax.plot(
                            fpr, tpr, label=MODELS[label][0], color=MODELS[label][1]
                        )
        if key == "mfnd":
            ax.set_xscale("log")
        else:
            # reference line
            ax.plot([0, 1], [0, 1], linestyle="--", color="k")
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    plt.savefig("figures/roc.eps", format="eps", bbox_inches="tight", pad_inches=0.1)
    plt.show()


def main(args):
    plot_roc(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args)
