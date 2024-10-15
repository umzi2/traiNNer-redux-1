import pyiqa
import torch
import torchvision.transforms.functional as tf
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

PATCH_SIZE_KADID = 384


@LOSS_REGISTRY.register()
class TOPIQLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, resize_input: bool = True) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.loss = pyiqa.create_metric(
            "topiq_fr",
            device="cuda" if torch.cuda.is_available() else "cpu",
            as_loss=True,
        )
        self.resize_input = resize_input

    def forward(self, x: Tensor, gt: Tensor) -> tuple[Tensor | None, Tensor | None]:
        if self.resize_input:
            if x.shape[2] != PATCH_SIZE_KADID or x.shape[3] != PATCH_SIZE_KADID:
                assert x.shape == gt.shape
                x = tf.resize(
                    x,
                    [PATCH_SIZE_KADID],
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                )

                gt = tf.resize(
                    gt,
                    [PATCH_SIZE_KADID],
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                )

        return (1 - self.loss(x, gt)) * self.loss_weight
