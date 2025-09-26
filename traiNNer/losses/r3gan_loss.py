"""R3GAN adversarial loss implementation.

This module implements the relativistic adversarial loss used by the
R3GAN baseline ("The GAN is dead; long live the GAN! A Modern Baseline GAN").
It adapts the generator and discriminator losses described in the reference
implementation to the traiNNer-redux training pipeline.
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor, autograd, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


def _identity(x: Tensor) -> Tensor:
    return x


@LOSS_REGISTRY.register()
class R3GANLoss(nn.Module):
    """Relativistic GAN loss with zero-centered gradient penalties.

    The implementation mirrors the public R3GAN repository while adapting the
    interface to match traiNNer-redux's loss construction. The loss computes a
    relativistic logistic adversarial objective paired with zero-centered
    gradient penalties on both real and generated samples.

    Args:
        loss_weight (float): Multiplier applied when the loss is used for the
            generator. The discriminator always uses a multiplier of ``1``.
        gamma (float): Weight for the zero-centered gradient penalty term. The
            final discriminator loss becomes ``adv + gamma / 2 * (r1 + r2)``.
        preprocess (Callable | None): Optional callable applied to the input
            samples before being fed to the discriminator. When ``None`` an
            identity transform is used. This mirrors the augmentation pipeline
            hook provided by the original R3GAN codebase.
    """

    def __init__(
        self,
        loss_weight: float,
        gamma: float = 1.0,
        preprocess: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.preprocess = preprocess or _identity

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover - handled explicitly
        raise RuntimeError(
            "R3GANLoss.forward is not used directly. Call ``generator_forward`` "
            "or ``discriminator_forward`` instead."
        )

    @staticmethod
    def _zero_centered_gradient_penalty(samples: Tensor, critics: Tensor) -> Tensor:
        gradient, = autograd.grad(
            outputs=critics.sum(), inputs=samples, create_graph=True
        )
        return gradient.square().sum(dim=[1, 2, 3])

    def generator_forward(
        self,
        net_d: nn.Module,
        fake_samples: Tensor,
        real_samples: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the generator loss.

        Args:
            net_d (nn.Module): Discriminator network.
            fake_samples (Tensor): Generated samples (requires gradient).
            real_samples (Tensor): Real samples used for the relativistic term.

        Returns:
            tuple: A tuple containing the scalar generator loss tensor and a
            dictionary with detached diagnostic tensors.
        """

        processed_fake = self.preprocess(fake_samples)
        with torch.no_grad():
            processed_real = self.preprocess(real_samples.detach())
            real_logits = net_d(processed_real)

        fake_logits = net_d(processed_fake)
        relativistic_logits = fake_logits - real_logits
        adversarial_loss = F.softplus(-relativistic_logits).mean()

        diagnostics = {
            "relativistic_logits": relativistic_logits.detach(),
            "fake_logits": fake_logits.detach(),
            "real_logits": real_logits.detach(),
        }

        return adversarial_loss, diagnostics

    def discriminator_forward(
        self,
        net_d: nn.Module,
        real_samples: Tensor,
        fake_samples: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the discriminator loss and auxiliary statistics.

        Args:
            net_d (nn.Module): Discriminator network.
            real_samples (Tensor): Real input images.
            fake_samples (Tensor): Generated images from the generator.

        Returns:
            dict[str, Tensor]: Dictionary containing the total loss and
            diagnostics. The ``total_loss`` entry should be used for the
            backward pass.
        """

        real_detached = real_samples.detach().requires_grad_(True)
        fake_detached = fake_samples.detach().requires_grad_(True)

        processed_real = self.preprocess(real_detached)
        processed_fake = self.preprocess(fake_detached)

        real_logits = net_d(processed_real)
        fake_logits = net_d(processed_fake)

        relativistic_logits = real_logits - fake_logits
        adversarial_loss = F.softplus(-relativistic_logits).mean()

        r1_penalty = self._zero_centered_gradient_penalty(
            real_detached, real_logits
        ).mean()
        r2_penalty = self._zero_centered_gradient_penalty(
            fake_detached, fake_logits
        ).mean()

        r1_contrib = 0.5 * self.gamma * r1_penalty
        r2_contrib = 0.5 * self.gamma * r2_penalty
        total_loss = adversarial_loss + r1_contrib + r2_contrib

        return {
            "total_loss": total_loss,
            "adversarial_loss": adversarial_loss,
            "r1_penalty": r1_penalty,
            "r2_penalty": r2_penalty,
            "r1_contrib": r1_contrib,
            "r2_contrib": r2_contrib,
            "relativistic_logits": relativistic_logits.detach(),
            "real_logits": real_logits.detach(),
            "fake_logits": fake_logits.detach(),
        }
