import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision

from loss import _eye_like
from dataset import MEAN, STD


def save_pairs(pairs, file, horizontal=True):
    pairs = pairs * torch.tensor(STD).view(-1, 1, 1) + torch.tensor(MEAN).view(-1, 1, 1)
    _, n, c, h, w = pairs.shape
    if horizontal:
        pairs = pairs.transpose(1, 0)
    _, nrow, c, h, w = pairs.shape
    pairs = pairs.reshape(-1, c, h, w)
    torchvision.utils.save_image(pairs, file, nrow=nrow)


def save_farthest_duplicates(labels, distances, pairs, file, n=4):
    duplicate_pairs = pairs[labels == 1]
    duplicate_dists = distances[labels == 1]
    duplicate_pairs = duplicate_pairs[np.argsort(duplicate_dists)].flip(0)
    save_pairs(duplicate_pairs[:n], file, False)


def save_closest_diff(labels, distances, pairs, file, n=4):
    diff_pairs = pairs[labels == 0]
    diff_dists = distances[labels == 0]
    diff_pairs = diff_pairs[np.argsort(diff_dists)]
    save_pairs(diff_pairs[:n], file, False)


def hard_mine(anchor_emb, same_emb, same_pairs, max_dist=10):
    dist_mat = torch.cdist(anchor_emb, same_emb)
    pos = torch.diag(dist_mat)

    dist_mat = dist_mat + _eye_like(dist_mat) * max_dist
    min_neg, idx = torch.min(dist_mat, 1)
    diff_pairs = torch.stack(
        (
            same_pairs[:, 0],  # anchors from same
            same_pairs[idx, 1],  # closest different
        ),
        dim=1,
    )
    dist = torch.cat((pos, min_neg))
    labels = torch.cat((torch.ones_like(pos), torch.zeros_like(min_neg)))
    pairs = torch.cat((same_pairs, diff_pairs))

    return dist, labels, pairs, dist_mat
