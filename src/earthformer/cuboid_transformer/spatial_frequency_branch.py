"""Forced Spectral Learning Branch for Earthformer.

Produces freq_delta that is added to encoder input features, changing QKV and
thus attention patterns. Not a bypassable residual - the addition happens before
the encoder, and the delta focuses on mid-to-high frequencies where
attention-based smoothing loses detail.

Key design decisions:
  - Soft band-pass (learnable radius-&gt;weight) replaces hard high-pass.
  - Influence decoupled from band selection: band_w controls "which band",
    influence controls "how much" independently.
  - aux_head provides direct gradient path to FFT branch.
  - smoothness_loss prevents spectral spikes without forcing uniformity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForcedSpectralBranch(nn.Module):
    """Forced spectral learning via frequency-domain feature modulation.

    Architecture:
      x -> rFFT2(H,W) -> |amp|, phase
        -> soft band-pass (learnable radius->weight)
        -> state-conditioned band gating (influence + band_w decoupled)
        -> irfft2 -> freq_delta
        -> aux_head -> aux_pred  (for direct gradient)
        -> smoothness_loss       (anti-spike)
    """

    def __init__(self, dim: int = 64, num_bands: int = 4,
                 num_groups: int = 4, state_hidden: int = 16,
                 in_len: int = 14, out_len: int = 7,
                 aux_dim: int = 1, beta: float = 0.3):
        """
        Args:
            dim:          feature channels.
            num_bands:    number of learnable frequency bands.
            num_groups:   channel groups for state conditioning.
            state_hidden: hidden dim of the state conditioner MLP.
            in_len:       input temporal length (14 days).
            out_len:      output temporal length (7 days).
            aux_dim:      output channels for aux prediction (1 for SSTA).
            beta:         fixed mixing weight for freq_delta in encoder input.
        """
        super().__init__()

        assert dim % num_groups == 0
        group_dim = dim // num_groups

        # ── Soft band-pass: radius -&gt; learnable per-frequency weight ──
        self.freq_weight_fn = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))

        # ── State conditioner (two decoupled heads) ──
        self.group_pools = nn.ModuleList([
            nn.Linear(group_dim, state_hidden // num_groups)
            for _ in range(num_groups)
        ])
        self.state_mlp = nn.Linear(state_hidden, num_bands)       # band selection
        self.influence_mlp = nn.Linear(state_hidden, num_bands)   # modulation strength

        # ── Spectral gating ──
        self.freq_profiles = nn.Parameter(torch.zeros(num_bands, 1, 1))
        self.channel_gates = nn.Parameter(torch.full((num_bands, dim), -2.0))

        # ── Auxiliary prediction head (direct gradient for FFT branch) ──
        self.aux_head = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=(in_len - out_len + 1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim // 4, aux_dim, kernel_size=1),
        )

        self.beta = beta
        self.dim = dim
        self.num_bands = num_bands
        self.num_groups = num_groups
        self.out_len = out_len
        self._profiles_expanded = None

        self._init_band_pass()

    def _init_band_pass(self):
        """Initialize freq_weight_fn so high frequencies start with higher weight."""
        # Set the last linear layer bias to produce sigmoid ~ 0.7 for r=1 (high freq)
        # sigmoid(x) ~ 0.7 -&gt; x ~ 0.85
        # For r_norm=0 (DC): sigmoid(bias) ~ 0.2 -&gt; bias ~ -1.39
        # For r_norm=1 (Nyquist): sigmoid(w*1 + b) ~ 0.7 -&gt; w + b ~ 0.85
        # So: b ~ -1.39, w ~ 2.24
        w = self.freq_weight_fn[-1].weight
        b = self.freq_weight_fn[-1].bias
        nn.init.constant_(w, 2.2)
        nn.init.constant_(b, -1.4)

    @torch.no_grad()
    def _ensure_profile(self, H: int, W_f: int, device: torch.device):
        if (self._profiles_expanded is not None
                and self._profiles_expanded.shape[-2:] == (H, W_f)):
            return self._profiles_expanded.to(device=device)
        p = self.freq_profiles
        p = F.interpolate(p.unsqueeze(0), size=(H, W_f),
                          mode='bilinear', align_corners=False)
        self._profiles_expanded = p.squeeze(0)
        return self._profiles_expanded

    def _get_radius(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        h_f = torch.fft.fftfreq(H, device=device)
        w_f = torch.fft.rfftfreq(W, device=device)
        hy, wx = torch.meshgrid(h_f, w_f, indexing="ij")
        return torch.sqrt(hy ** 2 + wx ** 2)  # (H, W_f)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, H, W, D) - features after InitialEncoder.
        Returns:
            freq_delta: (B, T, H, W, D) - mid-to-high frequency residual.
            aux_pred:   (B, T_out, H, W, C_out) - auxiliary SST prediction.
            loss_smooth: scalar - smoothness regularization.
        """
        B, T, H, W, D = x.shape
        W_f = W // 2 + 1

        # ── 2D spatial rFFT ──
        X_f = torch.fft.rfft2(x.float(), dim=(-3, -2))           # (B,T,H,W_f,D) complex
        amp = torch.abs(X_f)                                       # (B,T,H,W_f,D)
        phase = torch.angle(X_f)

        # ── Soft band-pass: learnable frequency selection ──
        r = self._get_radius(H, W, x.device)                      # (H, W_f)
        r_norm = r / (r.max() + 1e-8)                             # normalized to [0, 1]
        freq_weight = torch.sigmoid(
            self.freq_weight_fn(r_norm.unsqueeze(-1)))             # (H, W_f, 1)
        freq_weight = freq_weight.squeeze(-1)                     # (H, W_f)

        # ── State conditioner ──
        groups = x.chunk(self.num_groups, dim=-1)
        z_parts = []
        for g, pool in zip(groups, self.group_pools):
            z = g.mean(dim=(1, 2, 3))
            z_parts.append(pool(z))
        context = torch.cat(z_parts, dim=-1)                       # (B, state_hidden)

        band_w = F.softmax(self.state_mlp(context), dim=-1)        # (B, K)
        influence = torch.sigmoid(self.influence_mlp(context))     # (B, K)
        band_w = band_w * (0.5 + influence)                        # in [0.25, 1.25]

        # ── Spectral gating ──
        profiles = self._ensure_profile(H, W_f, x.device)          # (K, H, W_f)
        gates = torch.sigmoid(self.channel_gates)                  # (K, D)

        logits = (band_w[:, :, None, None, None]                   # (B, K, 1, 1, 1)
                  * profiles[None, :, :, :, None]                  # (1, K, H, W_f, 1)
                  * gates[None, :, None, None, :])                 # (1, K, 1, 1, D)
        attn = F.softmax(logits, dim=1)                            # (B, K, H, W_f, D)
        spectral_weight = (attn * profiles[None, :, :, :, None])\
            .sum(dim=1)                                            # (B, H, W_f, D)

        # Apply both freq_weight (soft band-pass) and spectral gating
        weight = spectral_weight.unsqueeze(1) \
            * freq_weight[None, None, :, :, None]                  # (B, 1, H, W_f, D)
        amp_enhanced = amp * weight                                  # (B, T, H, W_f, D)

        # ── Reconstruct -&gt; freq_delta (mid-high frequency residual) ──
        X_enhanced = amp_enhanced * torch.exp(1j * phase)
        freq_delta = torch.fft.irfft2(X_enhanced, s=(H, W), dim=(-3, -2))
        freq_delta = freq_delta.to(x.dtype)                        # (B, T, H, W, D)

        # ── Auxiliary prediction ──
        aux_in = freq_delta.permute(0, 4, 1, 2, 3)                # (B, D, T, H, W)
        aux_pred = self.aux_head(aux_in)                           # (B, C_out, T_out, H, W)
        aux_pred = aux_pred.permute(0, 2, 3, 4, 1)                # (B, T_out, H, W, C_out)

        # ── Smoothness loss (anti-spike) ──
        loss_smooth = _spectral_smoothness(freq_delta)

        return freq_delta, aux_pred, loss_smooth


def _spectral_smoothness(freq_delta):
    """Penalize isolated spectral spikes. Does NOT enforce uniform energy."""
    X_f = torch.fft.rfft2(freq_delta.float(), dim=(-3, -2))
    amp = torch.abs(X_f).mean(dim=(0, 1, -1))                     # (H, W_f)
    if amp.numel() < 2:
        return torch.tensor(0.0, device=freq_delta.device)
    h_diff = (amp[1:, :] - amp[:-1, :]).pow(2).mean()
    w_diff = (amp[:, 1:] - amp[:, :-1]).pow(2).mean()
    return (h_diff + w_diff) * 0.01
