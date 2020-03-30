import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics

from main import RetrievalModule
from dataset import TestDataset
import utils


def test_step(model, batch):
    anchor, same = batch
    anchor_emb, same_emb = model(anchor, same)
    pair = torch.stack([anchor, same], dim=1)
    return {"anchor_emb": anchor_emb, "same_emb": same_emb, "pair": pair}


@torch.no_grad()
def test(model, loader):
    test_out = [test_step(model, batch) for batch in loader]
    anchor_emb = torch.cat([out["anchor_emb"] for out in test_out])
    same_emb = torch.cat([out["same_emb"] for out in test_out])
    pairs = torch.cat([out["pair"] for out in test_out])
    return anchor_emb, same_emb, pairs


def main(args):
    model = RetrievalModule.load_from_checkpoint(args.checkpoint)
    test_set = TestDataset(args.test_path, filetype=args.filetype)
    test_loader = DataLoader(
        test_set, num_workers=4, batch_size=model.hparams.batch_size
    )

    # aggregate all embeddings and image pairs
    anchor_emb, same_emb, pairs = test(model, test_loader)

    # mine positive and hardest negative pairs
    distances, labels, pairs, dist_mat = utils.hard_mine(anchor_emb, same_emb, pairs)

    distances, labels = distances.numpy(), labels.numpy()
    print("AUC", metrics.roc_auc_score(labels, 1 / (distances + 1e-8)))

    # save FPR/TPR for figures
    if args.save_roc is not None:
        fpr, tpr, _ = metrics.roc_curve(labels, 1 / (distances + 1e-8))
        roc = np.stack([fpr, tpr])
        np.save(args.save_roc, roc)

    # save farthest duplicate images
    if args.save_farthest_dup is not None:
        utils.save_farthest_duplicates(labels, distances, pairs, args.save_farthest_dup)

    # save closets different images
    if args.save_closest_diff is not None:
        utils.save_closest_diff(labels, distances, pairs, args.save_closest_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--filetype", default=".tif")
    parser.add_argument("--save_roc", help="file to save roc")
    parser.add_argument("--save_farthest_dup", help="file to save farthest duplicates")
    parser.add_argument("--save_closest_diff", help="file to save closest different")

    parser.add_argument("test_path")
    parser.add_argument("checkpoint")

    args = parser.parse_args()
    main(args)
