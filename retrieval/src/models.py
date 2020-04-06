import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from torch.nn.parameter import Parameter
from torchvision import models


class Retrieval(nn.Module):
    def __init__(
        self, base="resnet50", out_dim=2048, pretrained=True, gemp=3, chunks=0
    ):
        """
        Construct BioMetric model.
        Parameters:
            out_dim (int): size of output embedding.
            pretrained (bool): use pretrained ResNet101 weights.
            gemp (float): initial p value for GeM pool.
            chunks (int): number of chuncks for gradient checkpointing,
                0 to disable.
        """
        super(Retrieval, self).__init__()

        assert base in ["resnet50", "vgg19"]

        self.store_grads = False
        self.chunks = chunks

        if base == "resnet50":
            # create resnet, remove final 2 layers (avgpool and fc)
            resnet = models.resnet50(pretrained=pretrained)
            self.base = nn.Sequential(*list(resnet.children())[:-2])

            # last bottleneck in resnet is 512 with expansion of 4
            self.fc = nn.Linear(512 * 4, out_dim)

        elif base == "vgg19":
            vgg = models.vgg19_bn(pretrained=pretrained)
            self.base = vgg.features

            self.fc = nn.Linear(512, out_dim)

        self.gem = GeneralizedMeanPoolingP(norm=gemp)

    def features(self, x):
        if self.chunks == 0:
            return self.base(x)
        modules = [module for _, module in self.base._modules.items()]
        return checkpoint_sequential(modules, self.chunks, x)

    def forward(self, x):
        x = self.features(x)

        x = self.gem(x)

        x = x.squeeze()
        x = self.fc(x)

        # L2 norm
        x = F.normalize(x, p=2, dim=-1)
        return x


#
# Below code is copied verbatim from:
# https://github.com/almazan/deep-image-retrieval/blob/master/dirtorch/nets/layers/pooling.py
#


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """
    Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(torch.ones(1) * norm)
