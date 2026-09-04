import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LoRALinear(nn.Module):
    """Frozen linear layer with a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        parameter_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, **parameter_kwargs)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **parameter_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        residual = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + residual * self.scale


def inject_finegrained_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
) -> list[str]:
    """Freeze A0 and add LoRA only to timestep, cross-attention, and modulation paths."""
    model.requires_grad_(False)
    target_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_timestep = name in {"t_embedder.mlp.0", "t_embedder.mlp.2"}
        is_cross_attention = ".cross_attn." in name
        is_block_modulation = (
            name.startswith("blocks.")
            and (name.endswith("adaLN_modulation.1") or name.endswith("text_gate"))
        )
        is_final_modulation = name in {
            "final_layer.adaLN_modulation.1",
            "final_layer.text_final",
        }
        if is_timestep or is_cross_attention or is_block_modulation or is_final_modulation:
            target_names.append(name)

    for name in target_names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        child_name = parts[-1]
        base = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        wrapped = LoRALinear(base, rank=int(rank), alpha=float(alpha))
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)
    return target_names


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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz, num_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, context_mask=None):
        bsz, num_tokens, channels = x.shape
        ctx_tokens = context.shape[1]
        q = self.q(x).reshape(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(
            bsz, ctx_tokens, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if context_mask is not None:
            if context_mask.shape != (bsz, ctx_tokens):
                raise ValueError(
                    f"Expected context mask {(bsz, ctx_tokens)}, got {tuple(context_mask.shape)}"
                )
            valid = context_mask[:, None, None, :].bool()
            attn = attn.masked_fill(~valid, torch.finfo(attn.dtype).min)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class FineGrainedDiTBlock(nn.Module):
    """Skeleton tokens query text tokens without rewriting the text condition."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        enable_self_attention=True,
        enable_cross_attention=True,
        **block_kwargs,
    ):
        super().__init__()
        self.enable_self_attention = bool(enable_self_attention)
        self.enable_cross_attention = bool(enable_cross_attention)
        self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_text = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.self_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))
        self.text_gate = nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        self.detail_norm_text = None
        self.detail_cross_attn = None
        self.detail_gate = None
        self.source_csv_norm_text = None
        self.source_llm_norm_text = None
        self.source_csv_cross_attn = None
        self.source_llm_cross_attn = None
        self.source_router_query = None
        self.source_router_condition = None

    def add_detail_residual(self, hidden_size, num_heads, gate_init=0.0, **block_kwargs):
        """Attach an optional, zero-gated text-detail cross-attention path."""
        self.detail_norm_text = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.detail_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.detail_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def add_dynamic_source_residual(self, hidden_size, num_heads, **block_kwargs):
        """Attach a zero-output CSV/LLM router conditioned on skeleton state."""
        self.source_csv_norm_text = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.source_llm_norm_text = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.source_csv_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.source_llm_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.source_router_query = nn.Linear(hidden_size, 1, bias=True)
        self.source_router_condition = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        x,
        text_tokens,
        c,
        fc_raw,
        inject_text=True,
        text_mask=None,
        detail_tokens=None,
        detail_mask=None,
        detail_reliability=None,
        source_csv_tokens=None,
        source_llm_tokens=None,
        source_csv_mask=None,
        source_llm_mask=None,
    ):
        params = self.adaLN_modulation(c) + self.text_gate(fc_raw)
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = params.chunk(9, dim=1)

        if self.enable_self_attention:
            x = x + gate_self.unsqueeze(1) * self.self_attn(
                modulate(self.norm_self(x), shift_self, scale_self)
            )
        cross_query = modulate(self.norm_cross(x), shift_cross, scale_cross)
        if self.enable_cross_attention and inject_text:
            if isinstance(text_tokens, tuple):
                if len(text_tokens) != 2:
                    raise ValueError(f"Balanced text fusion expects two views, got {len(text_tokens)}")
                view_masks = text_mask if isinstance(text_mask, tuple) else (None, None)
                cross_out = 0.5 * (
                    self.cross_attn(cross_query, self.norm_text(text_tokens[0]), view_masks[0])
                    + self.cross_attn(cross_query, self.norm_text(text_tokens[1]), view_masks[1])
                )
            else:
                cross_out = self.cross_attn(
                    cross_query,
                    self.norm_text(text_tokens),
                    text_mask,
                )
            x = x + gate_cross.unsqueeze(1) * cross_out
        if inject_text and self.detail_cross_attn is not None and detail_tokens is not None:
            detail_out = self.detail_cross_attn(
                cross_query,
                self.detail_norm_text(detail_tokens),
                detail_mask,
            )
            detail_scale = torch.tanh(self.detail_gate)
            if detail_reliability is not None:
                detail_scale = detail_scale * detail_reliability.reshape(-1, 1, 1)
            x = x + detail_scale * detail_out
        if inject_text and self.source_csv_cross_attn is not None and source_csv_tokens is not None:
            csv_out = self.source_csv_cross_attn(
                cross_query,
                self.source_csv_norm_text(source_csv_tokens),
                source_csv_mask,
            )
            llm_out = self.source_llm_cross_attn(
                cross_query,
                self.source_llm_norm_text(source_llm_tokens),
                source_llm_mask,
            )
            source_gate = torch.sigmoid(
                self.source_router_query(cross_query)
                + self.source_router_condition(c).unsqueeze(1)
            )
            x = x + source_gate * csv_out + (1.0 - source_gate) * llm_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )
        return x


class FineGrainedFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.text_final = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x, c, fc_raw, inject_text=True):
        if not inject_text:
            fc_raw = torch.zeros_like(fc_raw)
        shift, scale = (self.adaLN_modulation(c) + self.text_final(fc_raw)).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiTFineGrained(nn.Module):
    """Fine-grained feature-token DiT for text-sensitive reconstruction scoring.

    The feature-path C2U task provides one 256-d skeleton feature token. This
    model splits that vector into multiple skeleton tokens, lets those tokens
    query frozen text tokens with cross-attention, then stitches the predicted
    token chunks back into the original 256-d feature.
    """

    def __init__(
        self,
        in_channels=256,
        cond_size=2048,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        feature_tokens=8,
        text_fusion="channel",
        use_text_mask=False,
        text_detail_layers=None,
        agreement_topk=8,
        agreement_threshold=0.6,
        agreement_temperature=0.15,
        agreement_detail_gate_init=0.0,
        agreement_zero_out_proj=False,
        enable_self_attention=True,
        enable_cross_attention=True,
        enable_global_text=True,
        text_injection_layers=None,
        text_injection_final=True,
    ):
        super().__init__()
        feature_tokens = int(feature_tokens)
        if feature_tokens < 1:
            raise ValueError("feature_tokens must be >= 1")
        if in_channels % feature_tokens != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by feature_tokens={feature_tokens}")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.feature_tokens = feature_tokens
        self.token_channels = in_channels // feature_tokens
        self.cond_size = int(cond_size)
        self.text_fusion = str(text_fusion).lower()
        self.use_text_mask = bool(use_text_mask)
        if self.text_fusion not in {
            "channel",
            "view_tokens",
            "balanced_view_tokens",
            "agreement_mid_residual",
            "dynamic_mid_residual",
        }:
            raise ValueError(f"Unsupported text_fusion: {text_fusion}")
        if self.text_fusion not in {
            "channel",
            "agreement_mid_residual",
            "dynamic_mid_residual",
        } and self.cond_size % 2 != 0:
            raise ValueError(f"{self.text_fusion} fusion requires equal CSV/LLM feature dimensions")
        if self.text_fusion in {"agreement_mid_residual", "dynamic_mid_residual"} and self.cond_size % 2 != 0:
            raise ValueError(f"{self.text_fusion} requires equal CSV/LLM feature dimensions")
        self.text_detail_layers = sorted(
            {int(layer) for layer in (text_detail_layers or [5, 6, 7, 8])}
        )
        if self.text_fusion not in {"agreement_mid_residual", "dynamic_mid_residual"}:
            self.text_detail_layers = []
        if any(layer < 1 or layer > int(depth) for layer in self.text_detail_layers):
            raise ValueError(f"text_detail_layers must be within [1, {depth}]")
        self.agreement_topk = max(1, int(agreement_topk))
        self.agreement_threshold = float(agreement_threshold)
        self.agreement_temperature = max(float(agreement_temperature), 1e-6)
        self.agreement_detail_gate_init = float(agreement_detail_gate_init)
        self.agreement_zero_out_proj = bool(agreement_zero_out_proj)
        self.enable_self_attention = bool(enable_self_attention)
        self.enable_cross_attention = bool(enable_cross_attention)
        self.enable_global_text = bool(enable_global_text)
        if text_injection_layers is None:
            text_injection_layers = range(1, int(depth) + 1)
        self.text_injection_layers = sorted({int(layer) for layer in text_injection_layers})
        if any(layer < 1 or layer > int(depth) for layer in self.text_injection_layers):
            raise ValueError(f"text_injection_layers must be within [1, {depth}]")
        self.text_injection_final = bool(text_injection_final)

        self.x_embedder = nn.Linear(self.token_channels, hidden_size, bias=True)
        self.x_pos_embed = nn.Parameter(torch.zeros(1, feature_tokens, hidden_size), requires_grad=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fc_embedder = nn.Sequential(
            nn.Linear(cond_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        if self.text_fusion in {"channel", "agreement_mid_residual", "dynamic_mid_residual"}:
            # Preserve the original module name and shape for A0 checkpoint compatibility.
            self.fl_embedder = nn.Linear(cond_size, hidden_size, bias=True)
        else:
            view_size = self.cond_size // 2
            self.csv_fl_embedder = nn.Linear(view_size, hidden_size, bias=True)
            self.llm_fl_embedder = nn.Linear(view_size, hidden_size, bias=True)
            self.csv_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
            self.llm_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.fl_pos_embed = nn.Parameter(torch.zeros(1, 35, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList(
            [
                FineGrainedDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_self_attention=self.enable_self_attention,
                    enable_cross_attention=self.enable_cross_attention,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FineGrainedFinalLayer(hidden_size, self.token_channels)
        self.initialize_weights()
        if self.text_fusion == "agreement_mid_residual":
            self._add_agreement_detail_residuals(hidden_size, num_heads)
        elif self.text_fusion == "dynamic_mid_residual":
            self._add_dynamic_source_residuals(hidden_size, num_heads)

    def _format_feature_tokens(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected feature tensor with rank 2 or 3, got {tuple(x.shape)}")
        if x.shape[1] == 1 and x.shape[2] == self.in_channels:
            return x.reshape(x.shape[0], self.feature_tokens, self.token_channels)
        if x.shape[1] == self.feature_tokens and x.shape[2] == self.token_channels:
            return x
        raise ValueError(
            "Expected features shaped (B, 1, in_channels) or "
            f"(B, {self.feature_tokens}, {self.token_channels}), got {tuple(x.shape)}"
        )

    def _format_text_tokens(self, fl, fc):
        if fl is None:
            if fc.dim() != 2:
                raise ValueError(f"Expected global text condition with shape (B, D), got {tuple(fc.shape)}")
            return fc.unsqueeze(1).expand(-1, self.fl_pos_embed.shape[1], -1)
        if fl.dim() == 2:
            return fl.unsqueeze(1).expand(-1, self.fl_pos_embed.shape[1], -1)
        if fl.dim() != 3:
            raise ValueError(f"Expected text condition with rank 2 or 3, got {tuple(fl.shape)}")
        return fl

    def _text_pos_embed(self, num_tokens):
        if num_tokens > self.fl_pos_embed.shape[1]:
            raise ValueError(
                f"text token length {num_tokens} exceeds learned position length {self.fl_pos_embed.shape[1]}"
            )
        return self.fl_pos_embed[:, :num_tokens, :]

    def _embed_text_tokens(self, fl):
        if fl.shape[-1] != self.cond_size:
            raise ValueError(
                f"Expected local text dimension {self.cond_size}, got {fl.shape[-1]}"
            )
        pos = self._text_pos_embed(fl.shape[1])
        if self.text_fusion in {"channel", "agreement_mid_residual", "dynamic_mid_residual"}:
            text_mask = fl.abs().sum(dim=-1).ne(0) if self.use_text_mask else None
            return self.fl_embedder(fl) + pos, text_mask

        csv_raw, llm_raw = fl.chunk(2, dim=-1)
        csv_mask = csv_raw.abs().sum(dim=-1).ne(0) if self.use_text_mask else None
        llm_mask = llm_raw.abs().sum(dim=-1).ne(0) if self.use_text_mask else None
        csv_tokens = self.csv_fl_embedder(csv_raw) + pos + self.csv_source_embed
        llm_tokens = self.llm_fl_embedder(llm_raw) + pos + self.llm_source_embed
        if self.text_fusion == "balanced_view_tokens":
            text_mask = (
                (csv_mask, llm_mask)
                if csv_mask is not None and llm_mask is not None
                else None
            )
            return (csv_tokens, llm_tokens), text_mask

        text_mask = (
            torch.cat([csv_mask, llm_mask], dim=1)
            if csv_mask is not None and llm_mask is not None
            else None
        )
        return torch.cat([csv_tokens, llm_tokens], dim=1), text_mask

    def _add_agreement_detail_residuals(self, hidden_size, num_heads):
        # Build and initialize the detail path without perturbing A0's random stream.
        rng_state = torch.get_rng_state()
        view_size = self.cond_size // 2
        self.agreement_csv_embedder = nn.Linear(view_size, hidden_size, bias=True)
        self.agreement_llm_embedder = nn.Linear(view_size, hidden_size, bias=True)
        self.agreement_csv_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.agreement_llm_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for layer_idx in self.text_detail_layers:
            self.blocks[layer_idx - 1].add_detail_residual(
                hidden_size,
                num_heads,
                gate_init=self.agreement_detail_gate_init,
            )

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.agreement_csv_embedder.apply(_basic_init)
        self.agreement_llm_embedder.apply(_basic_init)
        for layer_idx in self.text_detail_layers:
            block = self.blocks[layer_idx - 1]
            block.detail_cross_attn.apply(_basic_init)
            if self.agreement_zero_out_proj:
                nn.init.constant_(block.detail_cross_attn.proj.weight, 0)
                nn.init.constant_(block.detail_cross_attn.proj.bias, 0)
        nn.init.normal_(self.agreement_csv_source_embed, std=0.02)
        nn.init.normal_(self.agreement_llm_source_embed, std=0.02)
        torch.set_rng_state(rng_state)

    def _agreement_detail_tokens(self, fl):
        if fl.dim() != 3:
            raise ValueError("agreement_mid_residual requires token-level text features")
        csv_raw, llm_raw = fl.chunk(2, dim=-1)
        csv_mask = csv_raw.abs().sum(dim=-1).ne(0)
        llm_mask = llm_raw.abs().sum(dim=-1).ne(0)
        if not bool(csv_mask.any(dim=1).all()) or not bool(llm_mask.any(dim=1).all()):
            raise ValueError("agreement_mid_residual requires non-empty CSV and LLM token views")

        csv_norm = F.normalize(csv_raw.float(), dim=-1)
        llm_norm = F.normalize(llm_raw.float(), dim=-1)
        similarity = torch.matmul(llm_norm, csv_norm.transpose(1, 2))
        similarity = similarity.masked_fill(~csv_mask[:, None, :], float("-inf"))
        llm_scores = similarity.max(dim=-1).values.masked_fill(~llm_mask, float("-inf"))
        topk = min(self.agreement_topk, int(llm_scores.shape[1]))
        top_scores, top_indices = torch.topk(llm_scores, k=topk, dim=1)
        top_valid = torch.gather(llm_mask, 1, top_indices)
        selected_llm = torch.gather(
            llm_raw,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, llm_raw.shape[-1]),
        )
        pos = self._text_pos_embed(fl.shape[1]).expand(fl.shape[0], -1, -1)
        selected_pos = torch.gather(
            pos,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, pos.shape[-1]),
        )
        csv_tokens = (
            self.agreement_csv_embedder(csv_raw)
            + pos
            + self.agreement_csv_source_embed
        )
        llm_tokens = (
            self.agreement_llm_embedder(selected_llm)
            + selected_pos
            + self.agreement_llm_source_embed
        )
        valid_scores = top_scores.masked_fill(~top_valid, 0.0)
        mean_score = valid_scores.sum(dim=1) / top_valid.sum(dim=1).clamp_min(1)
        reliability = torch.sigmoid(
            (mean_score - self.agreement_threshold) / self.agreement_temperature
        ).to(dtype=csv_tokens.dtype)
        return (
            torch.cat([csv_tokens, llm_tokens], dim=1),
            torch.cat([csv_mask, top_valid], dim=1),
            reliability,
        )

    def _add_dynamic_source_residuals(self, hidden_size, num_heads):
        # Preserve all A0 parameters and the subsequent training RNG sequence.
        rng_state = torch.get_rng_state()
        view_size = self.cond_size // 2
        self.dynamic_csv_embedder = nn.Linear(view_size, hidden_size, bias=True)
        self.dynamic_llm_embedder = nn.Linear(view_size, hidden_size, bias=True)
        self.dynamic_csv_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dynamic_llm_source_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for layer_idx in self.text_detail_layers:
            self.blocks[layer_idx - 1].add_dynamic_source_residual(hidden_size, num_heads)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.dynamic_csv_embedder.apply(_basic_init)
        self.dynamic_llm_embedder.apply(_basic_init)
        for layer_idx in self.text_detail_layers:
            block = self.blocks[layer_idx - 1]
            block.source_csv_cross_attn.apply(_basic_init)
            block.source_llm_cross_attn.apply(_basic_init)
            nn.init.constant_(block.source_csv_cross_attn.proj.weight, 0)
            nn.init.constant_(block.source_csv_cross_attn.proj.bias, 0)
            nn.init.constant_(block.source_llm_cross_attn.proj.weight, 0)
            nn.init.constant_(block.source_llm_cross_attn.proj.bias, 0)
            nn.init.constant_(block.source_router_query.weight, 0)
            nn.init.constant_(block.source_router_query.bias, 0)
            nn.init.constant_(block.source_router_condition.weight, 0)
            nn.init.constant_(block.source_router_condition.bias, 0)
        nn.init.normal_(self.dynamic_csv_source_embed, std=0.02)
        nn.init.normal_(self.dynamic_llm_source_embed, std=0.02)
        torch.set_rng_state(rng_state)

    def _dynamic_source_tokens(self, fl):
        if fl.dim() != 3:
            raise ValueError("dynamic_mid_residual requires token-level text features")
        csv_raw, llm_raw = fl.chunk(2, dim=-1)
        csv_mask = csv_raw.abs().sum(dim=-1).ne(0)
        llm_mask = llm_raw.abs().sum(dim=-1).ne(0)
        if not bool(csv_mask.any(dim=1).all()) or not bool(llm_mask.any(dim=1).all()):
            raise ValueError("dynamic_mid_residual requires non-empty CSV and LLM token views")
        pos = self._text_pos_embed(fl.shape[1])
        csv_tokens = (
            self.dynamic_csv_embedder(csv_raw)
            + pos
            + self.dynamic_csv_source_embed
        )
        llm_tokens = (
            self.dynamic_llm_embedder(llm_raw)
            + pos
            + self.dynamic_llm_source_embed
        )
        return csv_tokens, llm_tokens, csv_mask, llm_mask

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.x_pos_embed, std=0.02)
        nn.init.normal_(self.fl_pos_embed, std=0.02)
        if self.text_fusion in {"view_tokens", "balanced_view_tokens"}:
            nn.init.normal_(self.csv_source_embed, std=0.02)
            nn.init.normal_(self.llm_source_embed, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_gate.weight, 0)
            nn.init.constant_(block.text_gate.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.text_final.weight, 0)
        nn.init.constant_(self.final_layer.text_final.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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
        x = self._format_feature_tokens(x)
        x = self.x_embedder(x) + self.x_pos_embed

        fl = self._format_text_tokens(fl, fc)
        text_tokens, text_mask = self._embed_text_tokens(fl)
        detail_tokens = None
        detail_mask = None
        detail_reliability = None
        source_csv_tokens = None
        source_llm_tokens = None
        source_csv_mask = None
        source_llm_mask = None
        if self.text_fusion == "agreement_mid_residual":
            detail_tokens, detail_mask, detail_reliability = self._agreement_detail_tokens(fl)
        elif self.text_fusion == "dynamic_mid_residual":
            (
                source_csv_tokens,
                source_llm_tokens,
                source_csv_mask,
                source_llm_mask,
            ) = self._dynamic_source_tokens(fl)
        t = self.t_embedder(t)
        fc_raw = self.fc_embedder(fc)
        global_fc = fc_raw if self.enable_global_text else torch.zeros_like(fc_raw)
        c = t + global_fc

        hidden_states = []
        hidden_layers = set(hidden_layers or [])
        for layer_idx, block in enumerate(self.blocks, start=1):
            inject_text = layer_idx in self.text_injection_layers
            block_fc = global_fc if inject_text else torch.zeros_like(global_fc)
            block_c = t + block_fc
            x = block(
                x,
                text_tokens,
                block_c,
                block_fc,
                inject_text=inject_text,
                text_mask=text_mask,
                detail_tokens=detail_tokens if layer_idx in self.text_detail_layers else None,
                detail_mask=detail_mask if layer_idx in self.text_detail_layers else None,
                detail_reliability=detail_reliability if layer_idx in self.text_detail_layers else None,
                source_csv_tokens=source_csv_tokens if layer_idx in self.text_detail_layers else None,
                source_llm_tokens=source_llm_tokens if layer_idx in self.text_detail_layers else None,
                source_csv_mask=source_csv_mask if layer_idx in self.text_detail_layers else None,
                source_llm_mask=source_llm_mask if layer_idx in self.text_detail_layers else None,
            )
            if clean_token_residuals is not None and layer_idx in clean_token_residuals:
                residual = clean_token_residuals[layer_idx]
                if residual.shape != x.shape:
                    raise ValueError(
                        f"Clean token residual at layer {layer_idx} has shape {tuple(residual.shape)}, "
                        f"expected {tuple(x.shape)}"
                    )
                x = x + residual
            if return_hidden and (not hidden_layers or layer_idx in hidden_layers):
                hidden_states.append(x)

        final_c = c if self.text_injection_final else t
        x = self.final_layer(
            x,
            final_c,
            global_fc,
            inject_text=self.text_injection_final,
        )
        x = x.reshape(x.shape[0], 1, self.in_channels)
        if return_hidden:
            return x, hidden_states
        return x

    @staticmethod
    def pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=1)
