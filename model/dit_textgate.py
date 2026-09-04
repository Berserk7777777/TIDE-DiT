"""DiTTextGate — DiT with per-layer decoupled text injection.

Architecture change vs. the original DiT (model/dit.py):

Original:
    c = t_embed + fc_embed          # t and text mixed into ONE conditioning vector
    Each CrossDiTBlock: adaLN params ← c  (text is a weak additive component)

This file:
    c      = t_embed + fc_embed     # kept for backward-compatible modulation path
    fc_raw = fc_embed               # pure text embedding, NO timestep component

    Each CrossDiTBlockTG has TWO adaLN paths:
      1. adaLN_modulation(c)        — original mixed path (handles timing / scale)
      2. text_gate(fc_raw)          — pure text path, ZERO-initialised

    The text_gate weights start at zero, so the model begins identical to the
    original DiT and gradually learns to use the text gate.  This means:
      - Training is stable from the first step.
      - Gradients from InfoNCE / DCR now have a direct, t-independent path to
        shift and scale every token in every layer — the text is no longer a
        weak additive side-channel.

FinalLayer likewise gains a text_final branch (also zero-initialised).
"""

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp


class MHSA(nn.Module):
    """Bidirectional multi-head self-attention shared with the original DiT."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm: bool = False, attn_drop: float = 0.,
                 proj_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm   = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm   = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.q_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_c = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj   = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        qkv   = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        B, N_c, _ = cond.shape
        qkv_c = self.qkv_c(cond).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c.unbind(0)
        q_c, k_c = self.q_norm_c(q_c), self.k_norm_c(k_c)

        q = torch.cat((q, q_c), dim=-2)
        k = torch.cat((k, k_c), dim=-2)
        v = torch.cat((v, v_c), dim=-2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N + N_c, C)

        x_out, cond_out = out[:, :N], out[:, N:]
        return self.proj_drop(self.proj(x_out)), self.proj_drop(self.proj_c(cond_out))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half  = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class CrossDiTBlockTG(nn.Module):
    """CrossDiT block with an additional zero-init text gate.

    adaLN params = adaLN_modulation(c) + text_gate(fc_raw)
                   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                   mixed (t + text)      pure text, no timestep

    Since text_gate is zero-initialised, the block starts as a standard
    CrossDiTBlock and the text gate is learned on top.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = MHSA(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp   = Mlp(in_features=hidden_size, hidden_features=mlp_hidden, act_layer=approx_gelu, drop=0)
        self.mlp_c = Mlp(in_features=hidden_size, hidden_features=mlp_hidden, act_layer=approx_gelu, drop=0)

        # Original mixed (t + text) adaLN path
        self.adaLN_modulation   = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_c = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Pure-text gate — decoupled from timestep, zero-initialised
        self.text_gate   = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_gate_c = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        nn.init.zeros_(self.text_gate.weight)
        nn.init.zeros_(self.text_gate.bias)
        nn.init.zeros_(self.text_gate_c.weight)
        nn.init.zeros_(self.text_gate_c.bias)

    def forward(self, x, cond, c, fc_raw):
        # Mixed path params
        mixed   = self.adaLN_modulation(c)    # (B, 6*D)
        mixed_c = self.adaLN_modulation_c(c)

        # Pure-text gate params (additive, starts at zero)
        tg   = self.text_gate(fc_raw)          # (B, 6*D)
        tg_c = self.text_gate_c(fc_raw)

        shift_msa,   scale_msa,   gate_msa,   shift_mlp,   scale_mlp,   gate_mlp   = (mixed   + tg  ).chunk(6, dim=1)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = (mixed_c + tg_c).chunk(6, dim=1)

        x_temp    = modulate(self.norm1(x),    shift_msa,   scale_msa)
        cond_temp = modulate(self.norm1_c(cond), shift_msa_c, scale_msa_c)

        x_temp, cond_temp = self.attn(x_temp, cond_temp)

        x    = x    + gate_msa.unsqueeze(1)   * x_temp
        cond = cond + gate_msa_c.unsqueeze(1) * cond_temp

        x    = x    + gate_mlp.unsqueeze(1)   * self.mlp(modulate(self.norm2(x),    shift_mlp,   scale_mlp))
        cond = cond + gate_mlp_c.unsqueeze(1) * self.mlp_c(modulate(self.norm2_c(cond), shift_mlp_c, scale_mlp_c))
        return x, cond


class FinalLayerTG(nn.Module):
    """FinalLayer with an additional zero-init text gate."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.text_final = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.text_final.weight)
        nn.init.zeros_(self.text_final.bias)

    def forward(self, x, c, fc_raw):
        params = self.adaLN_modulation(c) + self.text_final(fc_raw)
        shift, scale = params.chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTTextGate(nn.Module):
    """DiT with per-layer decoupled text injection (text gate).

    Drop-in replacement for DiT.  Accepts identical constructor arguments and
    produces identical output tensor shapes.  The extra parameters (text_gate*,
    text_final) are zero-initialised so the model starts from the same
    effective function as the original DiT.

    Usage in config:
        model_type: text_gate    # triggers DiTTextGate instead of DiT
    """

    def __init__(self, in_channels=256, cond_size=2048,
                 hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads   = num_heads

        self.x_embedder  = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder  = TimestepEmbedder(hidden_size)
        self.fc_embedder = nn.Sequential(
            nn.Linear(cond_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.fl_embedder    = nn.Linear(cond_size, hidden_size, bias=True)
        self.fl_pos_embed   = nn.Parameter(torch.zeros(1, 35, hidden_size), requires_grad=True)

        self.blocks      = nn.ModuleList([
            CrossDiTBlockTG(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayerTG(hidden_size, in_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.fl_pos_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init original adaLN output projections
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight,   0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,     0)
            nn.init.constant_(block.adaLN_modulation_c[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_c[-1].bias,   0)
            # text_gate already zero-init in CrossDiTBlockTG.__init__

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)
        # text_final already zero-init in FinalLayerTG.__init__

    def _format_text_tokens(self, fl, fc):
        if fl is None:
            if fc.dim() != 2:
                raise ValueError(f"Expected global text condition (B, D), got {tuple(fc.shape)}")
            return fc.unsqueeze(1).expand(-1, self.fl_pos_embed.shape[1], -1)
        if fl.dim() == 2:
            return fl.unsqueeze(1).expand(-1, self.fl_pos_embed.shape[1], -1)
        if fl.dim() != 3:
            raise ValueError(f"Unexpected text condition rank: {tuple(fl.shape)}")
        return fl

    def forward(
        self,
        x,
        t,
        fc,
        fl,
        return_hidden=False,
        hidden_layers=None,
        clean_token_residuals=None,
    ):
        # Kept for a uniform trainer interface; this backbone has no CleanTIDE injection path.
        del clean_token_residuals
        x  = self.x_embedder(x)                          # (B, M_x, D)
        fl = self._format_text_tokens(fl, fc)
        fl = self.fl_embedder(fl) + self.fl_pos_embed    # (B, M_l, D)

        t_emb  = self.t_embedder(t)                      # (B, D)
        fc_raw = self.fc_embedder(fc)                    # (B, D)  — pure text, no t
        c      = t_emb + fc_raw                          # (B, D)  — mixed, for original adaLN

        hidden_states  = []
        hidden_layers  = set(hidden_layers or [])
        for layer_idx, block in enumerate(self.blocks, start=1):
            x, fl = block(x, fl, c, fc_raw)             # pass both c and fc_raw
            if return_hidden and (not hidden_layers or layer_idx in hidden_layers):
                hidden_states.append(x)

        x = self.final_layer(x, c, fc_raw)
        if return_hidden:
            return x, hidden_states
        return x

    @staticmethod
    def pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=1)
