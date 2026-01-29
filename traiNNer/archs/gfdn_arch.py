from collections.abc import Sequence
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import Tensor

from traiNNer.utils.registry import ARCH_REGISTRY


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(
        tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:]
    )
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


class ConvNXC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        gain1: int = 2,
        s: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.use_bias = bias
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.c_out = c_out
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )

        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        self.padding = (pad_h, pad_w)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=kernel_size,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )
        self.weight = nn.Parameter(
            torch.zeros(c_out, c_in, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(c_out))
        else:
            self.register_parameter("bias", None)

        nn.init.trunc_normal_(self.sk.weight, std=0.02)

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()

        kh, kw = self.kernel_size
        pad_h = kh - 1
        pad_w = kw - 1

        w = (
            F.conv2d(
                w1.flip(2, 3).permute(1, 0, 2, 3),
                w2,
                padding=(pad_h, pad_w),
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(
                w.flip(2, 3).permute(1, 0, 2, 3),
                w3,
                padding=0,
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        sk_w = self.sk.weight.data.clone().detach()

        if self.use_bias:
            b1 = self.conv[0].bias.data.clone().detach()
            b2 = self.conv[1].bias.data.clone().detach()
            b3 = self.conv[2].bias.data.clone().detach()
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()

        kh, kw = self.kernel_size
        h_pixels_to_pad = (kh - 1) // 2
        w_pixels_to_pad = (kw - 1) // 2

        sk_w = F.pad(
            sk_w,
            [w_pixels_to_pad, w_pixels_to_pad, h_pixels_to_pad, h_pixels_to_pad],
        )

        self.weight_concat = self.weight_concat + sk_w
        self.weight.data = self.weight_concat

        if self.use_bias:
            self.bias_concat = self.bias_concat + sk_b
            self.bias.data = self.bias_concat

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
        out = self.conv(x_pad) + self.sk(x)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Вычисляем padding на основе kernel_size
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = F.conv2d(
                x,
                self.weight,
                self.bias if self.use_bias else None,
                stride=self.stride,
                padding=self.padding,
            )
        return out


class InceptionConv2d(nn.Module):
    """Inception convolution"""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 7,  # 11
        band_kernel_size: int = 11,  # 17
        branch_ratio: float = 0.25,  # 0.25
        rep: bool = False,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.conv_hw = (
            ConvNXC(gc, gc, (square_kernel_size, square_kernel_size))
            if rep
            else nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2)
        )
        self.conv_w = (
            ConvNXC(gc, gc, (1, band_kernel_size))
            if rep
            else nn.Conv2d(
                gc,
                gc,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
            )
        )
        self.conv_h = (
            ConvNXC(gc, gc, (band_kernel_size, 1))
            if rep
            else nn.Conv2d(
                gc,
                gc,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
            )
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.conv_hw(x_hw), self.conv_w(x_w), self.conv_h(x_h)),
            dim=1,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))
        self.rms = dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True).mul(self.rms).add(self.eps)
        x = x / norm_x
        return x.mul(self.scale[:, None, None]).add(self.offset[:, None, None])


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        square_kernel_size: int = 11,
        band_kernel_size: int = 17,
        rep: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.norm = RMSNorm(dim)
        self.fc1 = (
            ConvNXC(dim, hidden * 2, (3, 3))
            if rep
            else nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        )
        self.act = nn.SiLU()
        self.split_indices = [hidden, hidden - dim, dim]
        self.conv = InceptionConv2d(dim, square_kernel_size, band_kernel_size, rep=rep)
        self.fc2 = (
            ConvNXC(hidden, dim, (1, 1)) if rep else nn.Conv2d(hidden, dim, 1, 1, 0)
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        return x + shortcut


@ARCH_REGISTRY.register()
class GILSR(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        scale: int = 4,
        dim: int = 48,
        rep: bool = True,
        expansion_ratios: Sequence[float] = (8 / 3, 8 / 3, 8 / 3, 8 / 3, 8 / 3, 8 / 3),
        square_kernel_size: int = 11,
        band_kernel_size: int = 17,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_to_dim = (
            ConvNXC(in_ch, dim, (3, 3)) if rep else nn.Conv2d(in_ch, dim, 3, 1, 1)
        )
        self.block_0 = GatedCNNBlock(
            dim, expansion_ratios[0], square_kernel_size, band_kernel_size, rep=rep
        )
        self.block_n = nn.Sequential(
            *[
                GatedCNNBlock(dim, er, square_kernel_size, band_kernel_size, rep=rep)
                for er in expansion_ratios[1:]
            ]
            + [ConvNXC(dim, dim, (3, 3)) if rep else nn.Conv2d(dim, dim, 3, 1, 1)]
        )
        self.conv_cat = (
            ConvNXC(dim * 3, dim, (1, 1)) if rep else nn.Conv2d(dim * 3, dim, 1)
        )
        self.dim_to_scale = nn.Sequential(
            nn.Conv2d(dim, in_ch * scale * scale, 3, 1, 1), nn.PixelShuffle(scale)
        )

        weight = ICNR(
            self.dim_to_scale[0].weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=scale,
        )
        self.dim_to_scale[0].weight.data.copy_(weight)  # initialize conv.weight

        self.shift = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.5, requires_grad=True)
        self.scale_norm = nn.Parameter(torch.ones(1, 3, 1, 1) / 6, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.shift) / self.scale_norm
        x = self.in_to_dim(x)
        x0 = self.block_0(x)
        x = self.conv_cat(torch.cat([x, self.block_n(x0), x0], dim=1))
        x = self.dim_to_scale(x)
        x = (x * self.scale_norm) + self.shift
        return x


@ARCH_REGISTRY.register()
def GILSR_s(
    expansion_ratios: Sequence[float] = (1.5, 2, 1.5, 2),
    square_kernel_size: int = 7,
    band_kernel_size: int = 11,
    **kwargs,
):
    return GILSR(
        expansion_ratios=expansion_ratios,
        square_kernel_size=square_kernel_size,
        band_kernel_size=band_kernel_size,
        **kwargs,
    )
