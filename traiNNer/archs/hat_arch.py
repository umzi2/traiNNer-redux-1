from spandrel.architectures.HAT import HAT
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def hat(**kwargs):
    return HAT(upscaling_factor=Config.get_scale(), **kwargs)


@SPANDREL_REGISTRY.register()
def hat_l(**kwargs):
    return HAT(upscale=Config.get_scale(),
               in_chans=3,
               window_size=16,
               compress_ratio=3,
               squeeze_factor=30,
               conv_scale=0.01,
               overlap_ratio=0.5,
               img_range=1.,
               depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               embed_dim=180,
               num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               mlp_ratio=2,
               upsampler="pixelshuffle",
               resi_connection="1conv",
               **kwargs)


@SPANDREL_REGISTRY.register()
def hat_m(**kwargs):
    return HAT(upscale=Config.get_scale(),
               in_chans=3,
               window_size=16,
               compress_ratio=3,
               squeeze_factor=30,
               conv_scale=0.01,
               overlap_ratio=0.5,
               img_range=1.,
               depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               embed_dim=180,
               num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               mlp_ratio=2,
               upsampler="pixelshuffle",
               resi_connection="1conv",
               **kwargs)


@SPANDREL_REGISTRY.register()
def hat_s(**kwargs):
    return HAT(upscale=Config.get_scale(),
               in_chans=3,
               img_size=64,
               window_size=16,
               compress_ratio=3,
               squeeze_factor=30,
               conv_scale=0.01,
               overlap_ratio=0.5,
               img_range=1.,
               depths=[6, 6, 6, 6, 6, 6],
               embed_dim=180,
               num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2,
               upsampler="pixelshuffle",
               resi_connection="1conv",
               **kwargs)
