"""State-conditioned adaptive spatial frequency gating for Earthformer.

Insert between InitialEncoder and enc_pos_embed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyBranch(nn.Module):
    """State-conditioned adaptive spatial frequency gating for Earthformer.

    Design:
        x -> rFFT2(H,W) -> |amp| -> state-conditioned band gating -> iFFT2 -> alpha * residual

    - Only processes amplitude; phase is preserved unchanged.
    - State conditioner: per-group GlobalAvgPool -> MLP -> sample-specific band weights.
    - Softmax over K bands provides bounded frequency conservation.

    Insert: x = self.initial_encoder(x)
            x = self.freq_branch(x)          # <-- here
            x = self.enc_pos_embed(x)
    """

    def __init__(self, dim: int = 64, num_bands: int = 4,
                 num_groups: int = 4, state_hidden: int = 16):
        """
        Args:
            dim:          feature channels (e.g. 64 after initial_encoder).
            num_bands:    number of learnable frequency bands.
            num_groups:   channel split groups for state conditioning.
            state_hidden: hidden dim of the state conditioner MLP.
        """
        super().__init__()

        assert dim % num_groups == 0
        group_dim = dim // num_groups

        # Per-group linear -> concat -> MLP -> softmax band weights
        self.group_pools = nn.ModuleList([
            nn.Linear(group_dim, state_hidden // num_groups)
            for _ in range(num_groups)
        ])
        self.state_mlp = nn.Linear(state_hidden, num_bands)

        self.freq_profiles = nn.Parameter(torch.zeros(num_bands, 1, 1))
        self.channel_gates = nn.Parameter(torch.full((num_bands, dim), -2.0))

        self.alpha = nn.Parameter(torch.zeros(1))

        self.dim = dim
        self.num_bands = num_bands
        self.num_groups = num_groups
        self._profiles_expanded = None

    @torch.no_grad()
    def _ensure_profile(self, H: int, W_f: int, device: torch.device):
        if (self._profiles_expanded is not None
                and self._profiles_expanded.shape[-2:] == (H, W_f)):
            return self._profiles_expanded.to(device=device)

        p = self.freq_profiles                                     # (K, 1, 1)
        p = F.interpolate(p.unsqueeze(0), size=(H, W_f),
                          mode='bilinear', align_corners=False)    # (1, K, H, W_f)
        self._profiles_expanded = p.squeeze(0)                     # (K, H, W_f)
        return self._profiles_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, D) - features after InitialEncoder.
        Returns:
            (B, T, H, W, D) - frequency-enhanced features, same shape.
        """
        B, T, H, W, D = x.shape
        W_f = W // 2 + 1

        # ---- 2D spatial rFFT ----
        X_f = torch.fft.rfft2(x.float(), dim=(-3, -2))            # (B,T,H,W_f,D) complex
        amp = torch.abs(X_f)                                       # (B,T,H,W_f,D)
        phase = torch.angle(X_f)

        # ---- State conditioner ----
        groups = x.chunk(self.num_groups, dim=-1)                  # each (B,T,H,W,D//G)
        z_parts = []
        for g, pool in zip(groups, self.group_pools):
            z = g.mean(dim=(1, 2, 3))                              # (B, D//G)
            z_parts.append(pool(z))                                # (B, H//G)
        context = torch.cat(z_parts, dim=-1)                       # (B, H_hidden)
        band_w = F.softmax(self.state_mlp(context), dim=-1)        # (B, K)

        # ---- Spectral gating ----
        profiles = self._ensure_profile(H, W_f, x.device)          # (K, H, W_f)
        gates = torch.sigmoid(self.channel_gates)                  # (K, D)

        logits = (band_w[:, :, None, None, None]                   # (B, K, 1, 1, 1)
                  * profiles[None, :, :, :, None]                  # (1, K, H, W_f, 1)
                  * gates[None, :, None, None, :])                 # (1, K, 1, 1, D)
        attn = F.softmax(logits, dim=1)                            # (B, K, H, W_f, D)
        weight = (attn * profiles[None, :, :, :, None]).sum(dim=1) # (B, H, W_f, D)

        amp_enhanced = amp * weight                                # (B, T, H, W_f, D)

        # ---- Reconstruct ----
        X_enhanced = amp_enhanced * torch.exp(1j * phase)
        freq_out = torch.fft.irfft2(X_enhanced, s=(H, W), dim=(-3, -2))
        freq_out = freq_out.to(x.dtype)

        return x + self.alpha * freq_out

    def entropy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy regularization to prevent single-band collapse.

        Call externally and add to total loss with weight ~1e-4.

        loss_entropy = log(K) + sum_k p_k * log(p_k)    (range: 0 ~ log(K))
        """
        groups = x.chunk(self.num_groups, dim=-1)
        z_parts = []
        for g, pool in zip(groups, self.group_pools):
            z = g.mean(dim=(1, 2, 3))
            z_parts.append(pool(z))
        context = torch.cat(z_parts, dim=-1)
        band_w = F.softmax(self.state_mlp(context), dim=-1)
        log_bw = torch.log(band_w + 1e-8)
        entropy = -(band_w * log_bw).sum(dim=-1).mean()
        K = self.num_bands
        return torch.log(torch.tensor(K, dtype=entropy.dtype,
                                      device=entropy.device)) - entropy
