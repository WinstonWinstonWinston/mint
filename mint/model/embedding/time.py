import torch
import math
from torch.nn import functional as F

class TimeEmbed(torch.nn.Module):
    r"""
    Sinusoidal time embedding for normalized scalar timesteps :math:`t\in[0,1]`.

    For an embedding size :math:`D`, let :math:`H=\lfloor D/2 \rfloor` and let
    :math:`m` be the frequency scale (``max_positions``). Define

    .. math::
        \tilde t \;=\; t\,m, \qquad
        \omega_k \;=\; m^{-\frac{k}{H-1}},\quad k=0,\dots,H-1,

    and the embedding

    .. math::
        \mathrm{emb}(t)
        \;=\;
        \big[
          \sin(\tilde t\,\omega_0),\ldots,\sin(\tilde t\,\omega_{H-1}),
          \cos(\tilde t\,\omega_0),\ldots,\cos(\tilde t\,\omega_{H-1})
        \big]\in\mathbb{R}^{2H}.

    If :math:`D` is odd, one trailing zero is appended so the final size is :math:`D`. From Fairseq. This matches the implementation in
    tensor2tensor, but differs slightly from the description in Section 3.5 of "Attention Is All You Need". 
    Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    :param timecfg:
        Config with ``embedding_dim`` (output size :math:`D`) and ``max_positions`` (the scale :math:`m`).
    :type timecfg: DictConfig
    """
    def __init__(self, cfg) -> None:    
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.max_positions = cfg.max_positions

    def forward(self, timesteps) -> torch.Tensor:
        r"""
        Build sinusoidal embeddings for a 1D batch of timesteps.

        :param timesteps:
            1D tensor of normalized times :math:`t\in[0,1]` with shape :math:`(B,)`.
            Internally scaled as :math:`\tilde t = t\,m` where :math:`m=\text{max\_positions}`.
        :type timesteps: torch.Tensor
        :return:
            Tensor of shape :math:`(B, D)` with concatenated
            :math:`[\sin(\tilde t\,\boldsymbol{\omega})\,\Vert\,\cos(\tilde t\,\boldsymbol{\omega})]`
            and optional zero-padding when :math:`D` is odd.
        :rtype: torch.Tensor
        """
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.max_positions
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb