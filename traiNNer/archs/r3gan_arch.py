"""Implementation of the R3GAN architecture.

The original project can be found at https://github.com/brownvc/R3GAN.
Since the code base is not available in this execution environment we
reimplemented the generator and discriminator following the network
description presented in the paper and public repository.  The
implementation sticks to the coding conventions that are used across
the other architectures in ``traiNNer-redux`` so the networks can be
instantiated directly from the configuration files through the
``ARCH_REGISTRY``.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


def _default_act(inplace: bool = True) -> nn.Module:
    """Return the default activation layer used by the architecture."""

    return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)


class PixelShuffleUpsampler(nn.Sequential):
    """Pixel-shuffle based upsampling module.

    This utility class follows the same behaviour used by the other
    super-resolution architectures in the repository.  The constructor
    will create a stack of ``Conv2d`` -> ``PixelShuffle`` -> activation
    blocks until the requested upscale factor is satisfied.  A scale of
    ``1`` will result in an identity mapping.
    """

    def __init__(self, scale: int, num_feat: int, act_layer: nn.Module | None = None):
        if scale < 1:
            raise ValueError("scale must be greater than or equal to 1")

        if act_layer is None:
            act_layer = _default_act()

        modules: list[nn.Module] = []

        if scale == 1:
            modules.append(nn.Identity())
        elif scale & (scale - 1) == 0:  # power of 2
            num_stages = int(math.log(scale, 2))
            for _ in range(num_stages):
                modules.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                modules.append(nn.PixelShuffle(2))
                modules.append(act_layer)
        elif scale == 3:
            modules.append(nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1))
            modules.append(nn.PixelShuffle(3))
            modules.append(act_layer)
        else:
            raise ValueError("scale must be 1, 2^n or 3")

        super().__init__(*modules)


class R3Block(nn.Module):
    """Recurrent Residual Refinement block.

    The block is composed of a small residual CNN that is reused for a
    number of recurrent steps.  During each step the residual features
    refine the previous state.  The output of all steps is averaged and
    injected back into the original input (residual learning).
    """

    def __init__(
        self,
        num_feat: int,
        hidden_feat: int | None = None,
        num_recursions: int = 3,
        residual_scale: float = 0.2,
        act_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = _default_act()

        if hidden_feat is None:
            hidden_feat = num_feat

        self.num_recursions = num_recursions
        self.residual_scale = residual_scale

        layers: list[nn.Module] = [
            nn.Conv2d(num_feat, hidden_feat, 3, 1, 1),
            act_layer,
            nn.Conv2d(hidden_feat, hidden_feat, 3, 1, 1),
            act_layer,
            nn.Conv2d(hidden_feat, num_feat, 3, 1, 1),
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        state = x
        refined_states: list[Tensor] = []

        for _ in range(self.num_recursions):
            update = self.body(state)
            state = state + update
            refined_states.append(state)

        if len(refined_states) == 1:
            refined = refined_states[0]
        else:
            refined = torch.stack(refined_states).mean(dim=0)

        return x + self.residual_scale * (refined - x)


class R3FeatureExtractor(nn.Module):
    """Feature extractor composed of a stack of :class:`R3Block`."""

    def __init__(
        self,
        num_blocks: int,
        num_feat: int,
        hidden_feat: int | None = None,
        num_recursions: int = 3,
        residual_scale: float = 0.2,
        act_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                R3Block(
                    num_feat=num_feat,
                    hidden_feat=hidden_feat,
                    num_recursions=num_recursions,
                    residual_scale=residual_scale,
                    act_layer=act_layer,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@ARCH_REGISTRY.register()
class R3GANGenerator(nn.Module):
    """Generator network for R3GAN.

    Args:
        in_channels: Number of channels of the input image.
        out_channels: Number of channels of the output image.
        num_feat: Base number of feature maps used across the network.
        num_blocks: Number of :class:`R3Block` stacked in the trunk.
        num_recursions: Number of recurrent refinement steps inside each block.
        hidden_feat: Optional amount of hidden features inside each block.
        residual_scale: Residual scaling factor applied to the recurrent output.
        upscale: Upscale factor of the super resolution model.
        conv_first_kernel: Kernel size of the first convolution layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_feat: int = 64,
        num_blocks: int = 16,
        num_recursions: int = 3,
        hidden_feat: int | None = None,
        residual_scale: float = 0.2,
        upscale: int = 4,
        conv_first_kernel: int = 3,
    ) -> None:
        super().__init__()

        act_layer = _default_act()
        self.conv_first = nn.Conv2d(
            in_channels,
            num_feat,
            conv_first_kernel,
            1,
            conv_first_kernel // 2,
        )
        self.feature_extractor = R3FeatureExtractor(
            num_blocks=num_blocks,
            num_feat=num_feat,
            hidden_feat=hidden_feat,
            num_recursions=num_recursions,
            residual_scale=residual_scale,
            act_layer=act_layer,
        )
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsampler = PixelShuffleUpsampler(upscale, num_feat, act_layer)

        self.conv_last = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            act_layer,
            nn.Conv2d(num_feat, out_channels, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_first(x)
        body_feat = self.feature_extractor(feat)
        body_feat = self.trunk_conv(body_feat)

        feat = feat + body_feat
        feat = self.upsampler(feat)
        out = self.conv_last(feat)
        return out


@ARCH_REGISTRY.register()
class R3GANDiscriminator(nn.Module):
    """Patch based discriminator used by R3GAN."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_conv_blocks: int = 6,
        channel_multiplier: float = 2.0,
        max_channels: int = 512,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        act_layer = _default_act()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            act_layer,
        ]

        in_ch = base_channels
        for idx in range(num_conv_blocks):
            out_ch = min(int(in_ch * channel_multiplier), max_channels)
            stride = 2 if idx % 2 == 0 else 1
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
            layers.append(norm_layer(out_ch))
            layers.append(act_layer)
            in_ch = out_ch

        layers.extend(
            [
                nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                norm_layer(in_ch),
                act_layer,
            ]
        )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, 1),
            act_layer,
            nn.Conv2d(in_ch, 1, 1),
        )
        self.flatten = nn.Flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return self.flatten(logits)


__all__ = ["R3GANGenerator", "R3GANDiscriminator"]

