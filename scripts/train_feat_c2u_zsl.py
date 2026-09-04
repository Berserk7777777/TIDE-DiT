from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from feeders.feeder import FeatureFeeder  # noqa: E402
from model.dit import DiT  # noqa: E402
from model.dit_finegrained import DiTFineGrained, inject_finegrained_lora  # noqa: E402
from model.dit_textgate import DiTTextGate  # noqa: E402


class EpisodicPrototypeGenerator(nn.Module):
    """Synthesize a held-out class prototype from semantic neighbours only."""

    def __init__(self, primitive_size: int, feature_size: int, hidden_size: int):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(primitive_size), nn.Linear(primitive_size, hidden_size))
        self.key = nn.Sequential(nn.LayerNorm(primitive_size), nn.Linear(primitive_size, hidden_size))
        self.residual = nn.Sequential(
            nn.LayerNorm(primitive_size),
            nn.Linear(primitive_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, feature_size),
        )

    def forward(
        self,
        target_primitives: torch.Tensor,
        support_primitives: torch.Tensor,
        support_prototypes: torch.Tensor,
        heldout_indices: torch.Tensor | None = None,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        query = F.normalize(self.query(target_primitives), dim=-1)
        key = F.normalize(self.key(support_primitives), dim=-1)
        logits = query @ key.transpose(0, 1) / max(float(temperature), 1e-6)
        if heldout_indices is not None:
            logits = logits.scatter(1, heldout_indices.view(-1, 1), float("-inf"))
        weights = torch.softmax(logits, dim=1)
        neighbour_prototype = weights @ support_prototypes
        # The residual is deliberately bounded so unseen synthesis remains
        # anchored in visual support prototypes rather than memorizing seen IDs.
        return neighbour_prototype + 0.10 * torch.tanh(self.residual(target_primitives))


class C2UAuxHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        text_size: int,
        feature_size: int = 256,
        proj_size: int = 256,
        fixed_text_space: bool = False,
    ):
        super().__init__()
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_size),
        )
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, proj_size),
        )
        if fixed_text_space:
            if proj_size != text_size:
                raise ValueError("fixed_text_space requires proj_size == text_size")
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_size),
                nn.Linear(text_size, proj_size),
            )
        self.primitive_feature_distribution: nn.Module | None = None
        self.episodic_proto_generator: EpisodicPrototypeGenerator | None = None
        self.frozen_gallery_feature_proj: nn.Module | None = None
        self.frozen_gallery_csv_proj: nn.Module | None = None
        self.frozen_gallery_llm_proj: nn.Module | None = None
        self.frozen_gallery_gate: nn.Module | None = None

    def configure_primitive_feature_distribution(
        self,
        primitive_size: int,
        feature_size: int,
        hidden_size: int,
    ) -> None:
        """Attach the C17 text-primitive to visual-feature distribution head."""
        if self.primitive_feature_distribution is not None:
            return
        self.primitive_feature_distribution = nn.Sequential(
            nn.LayerNorm(primitive_size),
            nn.Linear(primitive_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * feature_size),
        )

    def project_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.hidden_proj(x), dim=-1)

    def project_feature(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.feature_proj(x), dim=-1)

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(x), dim=-1)

    def configure_frozen_gallery_matcher(
        self,
        feature_size: int,
        text_size: int,
        proj_size: int,
    ) -> None:
        """Attach an efficient two-view matcher over frozen feature/text inputs."""
        if self.frozen_gallery_feature_proj is not None:
            return
        if int(text_size) % 2 != 0:
            raise ValueError("F1 frozen-gallery matching requires concatenated CSV/LLM text features")
        text_view_size = int(text_size) // 2
        self.frozen_gallery_feature_proj = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, proj_size),
            nn.SiLU(),
            nn.Linear(proj_size, proj_size),
        )
        self.frozen_gallery_csv_proj = nn.Sequential(
            nn.LayerNorm(text_view_size),
            nn.Linear(text_view_size, proj_size),
            nn.SiLU(),
            nn.Linear(proj_size, proj_size),
        )
        self.frozen_gallery_llm_proj = nn.Sequential(
            nn.LayerNorm(text_view_size),
            nn.Linear(text_view_size, proj_size),
            nn.SiLU(),
            nn.Linear(proj_size, proj_size),
        )
        self.frozen_gallery_gate = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, proj_size),
            nn.SiLU(),
            nn.Linear(proj_size, 1),
        )

    def frozen_gallery_match_logits(
        self,
        features: torch.Tensor,
        pooled_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return higher-is-better visual/text scores for one fixed candidate gallery."""
        if (
            self.frozen_gallery_feature_proj is None
            or self.frozen_gallery_csv_proj is None
            or self.frozen_gallery_llm_proj is None
            or self.frozen_gallery_gate is None
        ):
            raise RuntimeError("F1 frozen-gallery matcher was not initialized")
        if features.ndim != 2 or pooled_text.ndim != 2:
            raise ValueError("F1 matcher expects rank-2 feature and pooled-text tensors")
        if pooled_text.shape[1] % 2 != 0:
            raise ValueError("F1 matcher requires even concatenated text dimension")
        csv_text, llm_text = pooled_text.chunk(2, dim=1)
        visual = F.normalize(self.frozen_gallery_feature_proj(features), dim=-1)
        csv = F.normalize(self.frozen_gallery_csv_proj(csv_text), dim=-1)
        llm = F.normalize(self.frozen_gallery_llm_proj(llm_text), dim=-1)
        csv_logits = visual @ csv.T
        llm_logits = visual @ llm.T
        csv_gate = torch.sigmoid(self.frozen_gallery_gate(features))
        return csv_gate * csv_logits + (1.0 - csv_gate) * llm_logits, csv_gate.squeeze(1)

    def primitive_feature_distribution_parameters(
        self,
        primitive_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.primitive_feature_distribution is None:
            raise RuntimeError("C17 primitive feature-distribution head was not initialized")
        mean, raw_log_variance = self.primitive_feature_distribution(primitive_targets).chunk(2, dim=-1)
        # Prevent individual class variances from hiding a poor semantic mean.
        log_variance = raw_log_variance.tanh() * 1.5 - 0.5
        return mean, log_variance

    def configure_episodic_proto_generator(
        self,
        primitive_size: int,
        feature_size: int,
        hidden_size: int,
    ) -> None:
        if self.episodic_proto_generator is None:
            self.episodic_proto_generator = EpisodicPrototypeGenerator(
                primitive_size,
                feature_size,
                hidden_size,
            )


class CleanTideNeutralBranch(nn.Module):
    """Clean-feature student for D5, independent of the conditional DiT.

    The branch never receives a timestep or a class condition.  During
    training it learns to predict neutral teacher token representations from
    the frozen clean skeleton feature; its per-stratum projectors are only
    used by the distillation objective.
    """

    def __init__(self, feature_size: int, hidden_size: int, feature_tokens: int, strata: int):
        super().__init__()
        self.feature_tokens = int(feature_tokens)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, feature_tokens * hidden_size),
        )
        self.token_norm = nn.LayerNorm(hidden_size)
        self.projectors = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(int(strata))]
        )
        # Each projection starts as identity.  This makes the representation
        # learner, rather than a random head, carry the initial alignment.
        for projector in self.projectors:
            nn.init.zeros_(projector.weight)
            nn.init.zeros_(projector.bias)

    def encode(self, clean_features: torch.Tensor) -> torch.Tensor:
        flat = clean_features.flatten(1)
        tokens = self.encoder(flat).view(-1, self.feature_tokens, self.hidden_size)
        return self.token_norm(tokens)

    def project(self, tokens: torch.Tensor, stratum_index: int) -> torch.Tensor:
        return tokens + self.projectors[int(stratum_index)](tokens)


class CleanTideResidualFFN(nn.Module):
    """Zero-initialized residual MLP used by the CleanDIFT-style D5b branch."""

    def __init__(self, hidden_size: int, multiplier: int = 2):
        super().__init__()
        inner = int(hidden_size) * max(1, int(multiplier))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class CleanTideProjectionStack(nn.Module):
    """Training-only timestep projection; identity at initialization."""

    def __init__(self, hidden_size: int, multiplier: int = 2, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [CleanTideResidualFFN(hidden_size, multiplier) for _ in range(max(1, int(depth)))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CleanTideTokenBlock(nn.Module):
    """Small clean-token transformer block, independent of timestep and text."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_multiplier: int = 2):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size)
        inner = hidden_size * max(1, int(mlp_multiplier))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm_attn(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class CleanTideMultiLayerStudent(nn.Module):
    """D5b clean student with multi-layer distillation and zero-gated injection."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        feature_tokens: int,
        num_heads: int,
        depth: int,
        alignment_layers: list[int],
        strata: int,
        projection_multiplier: int = 2,
        projection_depth: int = 3,
    ):
        super().__init__()
        if feature_size % feature_tokens != 0:
            raise ValueError("CleanTIDE feature_size must be divisible by feature_tokens")
        self.feature_size = int(feature_size)
        self.feature_tokens = int(feature_tokens)
        self.hidden_size = int(hidden_size)
        self.alignment_layers = tuple(sorted({int(layer) for layer in alignment_layers}))
        self.input_proj = nn.Linear(feature_size // feature_tokens, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, feature_tokens, hidden_size))
        self.blocks = nn.ModuleList(
            [
                CleanTideTokenBlock(hidden_size, num_heads, mlp_multiplier=projection_multiplier)
                for _ in range(int(depth))
            ]
        )
        self.output_norm = nn.ModuleDict(
            {str(layer): nn.LayerNorm(hidden_size) for layer in self.alignment_layers}
        )
        self.projection_heads = nn.ModuleDict(
            {
                f"{layer}_{stratum}": CleanTideProjectionStack(
                    hidden_size,
                    multiplier=projection_multiplier,
                    depth=projection_depth,
                )
                for layer in self.alignment_layers
                for stratum in range(int(strata))
            }
        )
        # These paths enter the primary conditional DiT. Their zero-initialized
        # final layers make D5b an exact C10 control at step zero.
        self.condition_heads = nn.ModuleDict(
            {str(layer): CleanTideResidualFFN(hidden_size, projection_multiplier) for layer in self.alignment_layers}
        )
        for head in self.condition_heads.values():
            nn.init.zeros_(head.fc2.weight)
            nn.init.zeros_(head.fc2.bias)
        nn.init.normal_(self.position, std=0.02)

    def encode(self, clean_features: torch.Tensor) -> dict[int, torch.Tensor]:
        tokens = clean_features.reshape(
            clean_features.shape[0], self.feature_tokens, self.feature_size // self.feature_tokens
        )
        x = self.input_proj(tokens) + self.position
        states: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks, start=1):
            x = block(x)
            if layer_idx in self.alignment_layers:
                states[layer_idx] = self.output_norm[str(layer_idx)](x)
        return states

    def project(self, tokens: torch.Tensor, layer: int, stratum_index: int) -> torch.Tensor:
        return self.projection_heads[f"{int(layer)}_{int(stratum_index)}"](tokens)

    def condition_residuals(self, states: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {
            int(layer): self.condition_heads[str(layer)](states[int(layer)]) - states[int(layer)]
            for layer in self.alignment_layers
        }


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path: str | Path) -> dict:
    with repo_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def make_loader(config: dict, split_key: str, shuffle: bool, max_samples: int = 0) -> DataLoader:
    args = dict(config[f"{split_key}_feeder_args"])
    args["path"] = str(repo_path(args["path"]))
    dataset = FeatureFeeder(**args)
    labels = np.asarray(dataset.y, dtype=np.int64)
    include_values = config.get(f"{split_key}_include_labels", None)
    exclude_values = config.get(f"{split_key}_exclude_labels", None)
    keep = np.ones(labels.shape[0], dtype=bool)
    if include_values is not None:
        include = np.asarray(include_values, dtype=np.int64).reshape(-1)
        keep &= np.isin(labels, include)
    if exclude_values is not None:
        exclude = np.asarray(exclude_values, dtype=np.int64).reshape(-1)
        keep &= ~np.isin(labels, exclude)
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError(f"{split_key} label filtering removed every sample")
    if max_samples and indices.size > max_samples:
        indices = indices[:max_samples]
    if indices.size != len(dataset):
        dataset = Subset(dataset, indices.tolist())
    drop_last = bool(shuffle and len(dataset) >= int(config["batch_size"]))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"] if shuffle else config["test_batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_worker", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def loader_label_array(loader: DataLoader) -> np.ndarray:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        labels = np.asarray(dataset.dataset.y, dtype=np.int64)
        return labels[np.asarray(dataset.indices, dtype=np.int64)]
    return np.asarray(dataset.y, dtype=np.int64)


def build_text_embed(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
    config: dict | None = None,
) -> torch.Tensor:
    config = config or {}
    text_mode = str(config.get("text_condition_mode", "clip_token"))

    def load_llm_prompts() -> list[str]:
        """Load the second, existing text view without synthesizing new prompts."""
        label_path_value = config.get("text_llm_labels_path")
        if label_path_value:
            label_path = repo_path(label_path_value)
            values = np.load(label_path, allow_pickle=True)
            if values.ndim != 1:
                raise ValueError(
                    "text_llm_labels_path must contain a one-dimensional prompt array, "
                    f"got {tuple(values.shape)} from {label_path}"
                )
            return [str(value) for value in values.tolist()]

        llm_path = repo_path(config.get("text_llm_path", "data/class_lists/ntu60_llm.txt"))
        return llm_path.read_text(encoding="utf-8").splitlines()

    def mask_clip_padding(text_embed: torch.Tensor, tokenizer=None) -> torch.Tensor:
        if not bool(config.get("mask_text_padding", False)):
            return text_embed
        if text_embed.dim() != 3 or text_embed.shape[1] != 36:
            raise ValueError(
                "Masked CSV/LLM conditioning requires 35 local CLIP tokens plus one pooled token, "
                f"got {tuple(text_embed.shape)}"
            )
        if text_embed.shape[-1] % 2 != 0:
            raise ValueError("Masked CSV/LLM conditioning requires an even concatenated text dimension")

        class_list_path = repo_path(config.get("text_class_list_path", "data/class_lists/ntu60.csv"))
        csv_prompts = pd.read_csv(class_list_path)["label"].astype(str).tolist()
        llm_prompts = load_llm_prompts()
        if len(csv_prompts) != text_embed.shape[0] or len(llm_prompts) != text_embed.shape[0]:
            raise ValueError(
                "Text prompt counts must match the cached class dimension: "
                f"csv={len(csv_prompts)}, llm={len(llm_prompts)}, cache={text_embed.shape[0]}"
            )

        if tokenizer is None:
            tokenizer_path = config.get("text_tokenizer_name_or_path", model_path)
            hf_token = HfFolder.get_token()
            hf_kwargs = {"token": hf_token} if hf_token else {}
            tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                subfolder="tokenizer",
                local_files_only=local_files_only,
                **hf_kwargs,
            )

        masks = []
        for prompts in (csv_prompts, llm_prompts):
            tokenized = tokenizer(
                [prompt.strip() for prompt in prompts],
                padding="max_length",
                max_length=35,
                truncation=True,
                return_tensors="pt",
            )
            masks.append(tokenized.attention_mask.to(device=text_embed.device, dtype=torch.bool))

        masked = text_embed.clone()
        half = masked.shape[-1] // 2
        for view_idx, attention_mask in enumerate(masks):
            start = view_idx * half
            end = start + half
            masked[:, :35, start:end].masked_fill_(~attention_mask.unsqueeze(-1), 0)
        return masked

    def select_clip_source(text_embed):
        source = str(config.get("clip_text_source", "concat")).lower()
        if source in {"concat", "both", "csv_llm"}:
            return text_embed
        if source not in {"csv", "csv_only", "llm", "llm_only"}:
            raise ValueError(f"Unsupported clip_text_source: {source}")
        if text_embed.shape[-1] % 2 != 0:
            raise ValueError(
                "CSV/LLM slicing requires a concatenated text dimension divisible by 2, "
                f"got {text_embed.shape[-1]}"
            )
        half = text_embed.shape[-1] // 2
        return text_embed[..., :half] if source in {"csv", "csv_only"} else text_embed[..., half:]

    if text_mode in {"precomputed_semantic", "semantic"}:
        semantic_dir = repo_path(config.get("semantic_feature_dir", "data/text_feats/ntu60/semantic_ntu60_llm_sd2"))
        semantic_files = config.get("semantic_feature_files")
        if semantic_files:
            if not isinstance(semantic_files, (list, tuple)) or not semantic_files:
                raise ValueError("semantic_feature_files must be a non-empty list")
            arrays = []
            for semantic_file in semantic_files:
                semantic_path = semantic_dir / str(semantic_file)
                if not semantic_path.exists():
                    raise FileNotFoundError(f"Semantic feature file does not exist: {semantic_path}")
                array = np.load(semantic_path).astype(np.float32)
                if array.ndim != 2:
                    raise ValueError(
                        f"FS-VAE semantic features must be rank-2 (classes, dim), got {array.shape} from {semantic_path}"
                    )
                arrays.append(array)
            class_count = arrays[0].shape[0]
            if any(array.shape[0] != class_count for array in arrays[1:]):
                raise ValueError("All semantic feature files must contain the same number of classes")
            text_embed = np.concatenate(arrays, axis=-1)
        else:
            semantic_file = config.get("semantic_feature_file", "concat.npy")
            semantic_path = semantic_dir / semantic_file
            text_embed = np.load(semantic_path).astype(np.float32)
        label_file = semantic_dir / config.get("semantic_label_file", "labels.npy")
        if label_file.exists():
            labels = np.load(label_file).astype(np.int64)
            if labels.shape[0] == text_embed.shape[0] and not np.array_equal(labels, np.arange(text_embed.shape[0])):
                order = np.argsort(labels)
                text_embed = text_embed[order]
        if bool(config.get("semantic_l2_normalize", False)):
            norms = np.linalg.norm(text_embed, axis=-1, keepdims=True)
            text_embed = text_embed / np.clip(norms, a_min=1e-12, a_max=None)
        return torch.from_numpy(text_embed).to(device=device, dtype=dtype)

    cache_path_value = config.get("clip_text_feature_path")
    if cache_path_value:
        cache_path = repo_path(cache_path_value)
        if cache_path.exists():
            text_embed = np.load(cache_path).astype(np.float32)
            text_embed = torch.from_numpy(text_embed).to(device=device, dtype=dtype)
            text_embed = mask_clip_padding(text_embed)
            return select_clip_source(text_embed)

    hf_token = HfFolder.get_token()
    hf_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        local_files_only=local_files_only,
        **hf_kwargs,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        local_files_only=local_files_only,
        **hf_kwargs,
    )
    text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    class_list_path = repo_path(config.get("text_class_list_path", "data/class_lists/ntu60.csv"))
    df = pd.read_csv(class_list_path)
    csv_prompts = df["label"].values.tolist()
    llm_prompts = load_llm_prompts()
    if len(csv_prompts) != len(llm_prompts):
        raise ValueError(
            "CSV and second-view prompt counts must match: "
            f"csv={len(csv_prompts)}, second_view={len(llm_prompts)}"
        )

    def encode_prompts(prompts: list[str]) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for prompt in prompts:
                text_inputs = tokenizer(
                    prompt.strip(),
                    padding="max_length",
                    max_length=35,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                embeds = text_encoder(text_inputs.input_ids)
                chunks.append(
                    torch.cat(
                        (embeds["last_hidden_state"], embeds["pooler_output"].unsqueeze(1)),
                        dim=1,
                    ).detach()
                )
        return torch.cat(chunks, dim=0)

    text_embed = torch.cat((encode_prompts(csv_prompts), encode_prompts(llm_prompts)), dim=-1).to(dtype=dtype)
    if cache_path_value:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, text_embed.detach().cpu().numpy().astype(np.float32))
    return select_clip_source(mask_clip_padding(text_embed, tokenizer=tokenizer))


def split_text_condition(text_embed: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    cond = text_embed[labels]
    if cond.dim() == 2:
        return cond, None
    if cond.dim() == 3:
        return cond[:, -1, :], cond[:, :-1, :]
    raise ValueError(f"Unsupported text condition rank: {tuple(cond.shape)}")


def pooled_text_embed(text_embed: torch.Tensor) -> torch.Tensor:
    if text_embed.dim() == 2:
        return text_embed
    if text_embed.dim() == 3:
        return text_embed[:, -1, :]
    raise ValueError(f"Unsupported text condition rank: {tuple(text_embed.shape)}")


def load_c23_geometry_distance(
    metric_path_value: str,
    text_embed: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Load the frozen seen-only C23 text metric as a class-distance matrix."""
    metric_path = repo_path(metric_path_value)
    payload = np.load(metric_path)
    text_mean = payload["text_mean"].astype(np.float32)
    text_components = payload["text_components"].astype(np.float32)
    factor = payload["factor"].astype(np.float32)
    scale = float(payload["scale"])
    pooled = pooled_text_embed(text_embed).detach().float().cpu().numpy()
    if pooled.shape[1] != text_mean.shape[0]:
        raise ValueError("C23 metric text dimension does not match the active text condition")
    coordinates = ((pooled - text_mean) @ text_components.T) @ factor
    distance = scale * np.square(
        coordinates[:, None, :] - coordinates[None, :, :]
    ).sum(axis=-1)
    return torch.from_numpy(distance.astype(np.float32)).to(device=device)


def load_c24_reliability(
    reliability_path_value: str,
    class_count: int,
    device: torch.device,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Load Beta-shrunk, seen-only geometry-teacher reliability by anchor class."""
    payload = json.loads(repo_path(reliability_path_value).read_text(encoding="utf-8"))
    accuracy = payload.get("per_class_agreement", {})
    counts = payload.get("per_class_counts", {})
    reliability = torch.full((class_count,), 0.5, device=device, dtype=torch.float32)
    for key, value in accuracy.items():
        class_id = int(key)
        if 0 <= class_id < class_count:
            count = float(counts.get(key, 0))
            reliability[class_id] = (count * float(value) + alpha) / (count + alpha + beta)
    return reliability


def text_similarity_stats(text_embed: torch.Tensor, labels: np.ndarray | list[int]) -> dict[str, float]:
    label_list = [int(item) for item in labels]
    pooled = pooled_text_embed(text_embed).detach().float()
    subset = pooled[label_list]
    subset = F.normalize(subset, dim=-1)
    sim = subset @ subset.T
    mask = ~torch.eye(len(label_list), device=sim.device, dtype=torch.bool)
    values = sim[mask]
    return {
        "min": float(values.min().detach().cpu()) if values.numel() else 0.0,
        "mean": float(values.mean().detach().cpu()) if values.numel() else 0.0,
        "max": float(values.max().detach().cpu()) if values.numel() else 0.0,
    }


def transform_text_embed(text_embed: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    mode = str(config.get("text_decorrelation", "none")).lower()
    wants_center = bool(config.get("text_center", False))
    wants_l2 = bool(config.get("text_l2_normalize", False))
    wants_rescale = "text_rescale" in config
    remove_topk = int(config.get("text_remove_topk", 0))
    if mode in {"none", "false", "0"} and not (wants_center or wants_l2 or wants_rescale or remove_topk > 0):
        return text_embed, {}

    center = bool(config.get("text_center", mode in {"center", "pca_remove", "pc_remove", "whiten"}))
    rescale = bool(config.get("text_rescale", True))
    pooled = pooled_text_embed(text_embed).detach().float()
    mean = pooled.mean(dim=0, keepdim=True)
    original_rms = pooled.pow(2).mean().sqrt().clamp_min(1e-6)

    transformed = text_embed.float()
    if center or mode in {"center", "pca_remove", "pc_remove", "whiten"}:
        mean_view = mean.view(1, -1) if transformed.dim() == 2 else mean.view(1, 1, -1)
        transformed = transformed - mean_view.to(device=transformed.device, dtype=transformed.dtype)

    if remove_topk > 0 or mode in {"pca_remove", "pc_remove", "whiten"}:
        topk = max(1, remove_topk if remove_topk > 0 else int(config.get("text_whiten_topk", 3)))
        centered = (pooled - mean).cpu()
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        pcs = vh[: min(topk, vh.shape[0])].to(device=transformed.device, dtype=transformed.dtype)
        flat = transformed.reshape(-1, transformed.shape[-1])
        flat = flat - (flat @ pcs.T) @ pcs
        transformed = flat.reshape_as(transformed)

    if bool(config.get("text_l2_normalize", False)):
        transformed = F.normalize(transformed, dim=-1) * original_rms
    elif rescale:
        new_pooled = pooled_text_embed(transformed)
        new_rms = new_pooled.pow(2).mean().sqrt().clamp_min(1e-6)
        transformed = transformed * (original_rms.to(transformed.device) / new_rms)

    return transformed.to(device=text_embed.device, dtype=text_embed.dtype), {
        "mode": mode,
        "remove_topk": float(remove_topk),
        "center": float(center),
    }


def build_scheduler(config: dict, local_files_only: bool) -> DDPMScheduler:
    model_path = config.get("pretrained_model_name_or_path", "sd2-community/stable-diffusion-2-1")
    hf_token = HfFolder.get_token()
    hf_kwargs = {"token": hf_token} if hf_token else {}
    scheduler_config = DDPMScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        local_files_only=local_files_only,
        **hf_kwargs,
    ).config
    scheduler_config["num_train_timesteps"] = int(config["num_steps"])
    scheduler_config["prediction_type"] = str(config.get("prediction_type", "sample"))
    return DDPMScheduler.from_config(scheduler_config)


def build_logsnr_timestep_buckets(
    scheduler: DDPMScheduler,
    num_buckets: int,
    start_timestep: int,
    end_timestep: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Partition valid diffusion timesteps into equal-width log-SNR intervals."""
    start = max(1, int(start_timestep))
    end = min(int(end_timestep), int(scheduler.config.num_train_timesteps) - 1)
    if end < start or num_buckets < 1:
        raise ValueError("D5b requires a non-empty valid timestep range and positive bucket count")
    indices = torch.arange(start, end + 1, device=device, dtype=torch.long)
    alpha = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[indices]
    logsnr = (alpha / (1.0 - alpha).clamp_min(1e-8)).log()
    edges = torch.linspace(logsnr.max(), logsnr.min(), int(num_buckets) + 1, device=device)
    buckets: list[torch.Tensor] = []
    for bucket_index in range(int(num_buckets)):
        upper, lower = edges[bucket_index], edges[bucket_index + 1]
        if bucket_index == int(num_buckets) - 1:
            mask = (logsnr <= upper) & (logsnr >= lower)
        else:
            mask = (logsnr <= upper) & (logsnr > lower)
        bucket = indices[mask]
        if bucket.numel() == 0:
            midpoint = 0.5 * (upper + lower)
            bucket = indices[(logsnr - midpoint).abs().argmin()].view(1)
        buckets.append(bucket)
    return buckets


def build_band_tempered_elbo_schedule(
    scheduler: DDPMScheduler,
    start_timestep: int,
    end_timestep: int,
    num_bands: int,
    power: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a finite, semantically balanced x0-ELBO sampling distribution.

    The exact DDPM x0 KL weight diverges near t=0.  For conditional ZSL this
    would erase the high-noise, text-dependent regime.  We therefore temper
    the KL weights and normalize their total mass within each contiguous SNR
    band.  The returned objective weights have mean one over valid timesteps.
    """
    if num_bands < 1:
        raise ValueError("elbo_snr_num_bands must be positive")
    if power <= 0.0:
        raise ValueError("elbo_snr_power must be positive")
    betas = scheduler.betas.detach().to(device=device, dtype=torch.float32)
    alphas_cumprod = scheduler.alphas_cumprod.detach().to(device=device, dtype=torch.float32)
    total_steps = int(betas.numel())
    start = max(1, min(int(start_timestep), total_steps - 1))
    end = max(start, min(int(end_timestep), total_steps - 1))
    valid = torch.arange(start, end + 1, device=device, dtype=torch.long)
    if valid.numel() < num_bands:
        raise ValueError("ELBO timestep range must contain at least elbo_snr_num_bands entries")
    previous_alpha = torch.cat((torch.ones(1, device=device), alphas_cumprod[:-1]))
    posterior_variance = betas * (1.0 - previous_alpha) / (1.0 - alphas_cumprod).clamp_min(1e-12)
    raw_weight = (
        previous_alpha * betas.square()
        / (2.0 * posterior_variance.clamp_min(1e-12) * (1.0 - alphas_cumprod).square().clamp_min(1e-12))
    )
    tempered = raw_weight[valid].clamp_min(1e-12).pow(float(power))
    objective = torch.zeros(total_steps, device=device, dtype=torch.float32)
    band_indices = torch.tensor_split(torch.arange(valid.numel(), device=device), int(num_bands))
    band_mass = float(valid.numel()) / float(num_bands)
    band_summary: list[dict] = []
    for positions in band_indices:
        steps = valid[positions]
        local = tempered[positions]
        objective[steps] = local / local.sum().clamp_min(1e-12) * band_mass
        band_summary.append(
            {
                "start": int(steps[0].item()),
                "end": int(steps[-1].item()),
                "mass": float(objective[steps].sum().item()),
            }
        )
    probabilities = objective / objective.sum().clamp_min(1e-12)
    metadata = {
        "start": start,
        "end": end,
        "num_bands": int(num_bands),
        "power": float(power),
        "raw_weight_min": float(raw_weight[valid].min().item()),
        "raw_weight_max": float(raw_weight[valid].max().item()),
        "objective_weight_min": float(objective[valid].min().item()),
        "objective_weight_max": float(objective[valid].max().item()),
        "bands": band_summary,
    }
    return probabilities, objective, metadata


def alpha_sigma_for_timesteps(
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alphas = scheduler.alphas_cumprod.to(device=like.device, dtype=like.dtype)[timesteps.long()]
    view_shape = (-1,) + (1,) * (like.dim() - 1)
    alpha = alphas.sqrt().view(view_shape)
    sigma = (1.0 - alphas).sqrt().view(view_shape)
    return alpha, sigma


def diffusion_training_target(
    prediction_type: str,
    sample: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
) -> torch.Tensor:
    prediction_type = prediction_type.lower()
    if prediction_type in {"sample", "x", "x_prediction"}:
        return sample
    if prediction_type in {"epsilon", "eps"}:
        return noise
    if prediction_type in {"v_prediction", "v"}:
        return scheduler.get_velocity(sample, noise, timesteps.long())
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def prediction_to_sample(
    prediction: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
) -> torch.Tensor:
    prediction_type = prediction_type.lower()
    if prediction_type in {"sample", "x", "x_prediction"}:
        return prediction
    alpha, sigma = alpha_sigma_for_timesteps(scheduler, timesteps, noisy)
    if prediction_type in {"epsilon", "eps"}:
        return (noisy - sigma * prediction) / alpha.clamp_min(1e-6)
    if prediction_type in {"v_prediction", "v"}:
        return alpha * noisy - sigma * prediction
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def reverse_x0_deterministic(
    model: DiT,
    noisy: torch.Tensor,
    reverse_timesteps: list[int],
    fc: torch.Tensor,
    fl: torch.Tensor | None,
    scheduler: DDPMScheduler,
    prediction_type: str,
    dtype: torch.dtype,
    cfg_scale: float = 1.0,
    null_fc: torch.Tensor | None = None,
    clean_token_residuals: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the final x0 estimate from a deterministic, class-conditional reverse path."""
    state = noisy
    prediction = None
    for step_idx, timestep in enumerate(reverse_timesteps):
        current_t = torch.full(
            (state.shape[0],), int(timestep), device=state.device, dtype=torch.long
        )
        prediction = model(
            state,
            current_t.to(dtype=dtype),
            fc,
            fl,
            clean_token_residuals=clean_token_residuals,
        )
        if cfg_scale > 1.0:
            if null_fc is None:
                raise ValueError("Classifier-free guidance requires a null condition")
            null_prediction = model(
                state,
                current_t.to(dtype=dtype),
                null_fc,
                None,
                clean_token_residuals=clean_token_residuals,
            )
            prediction = (1.0 + cfg_scale) * prediction - cfg_scale * null_prediction
        pred_sample = prediction_to_sample(
            prediction,
            state,
            current_t,
            scheduler,
            prediction_type,
        )
        if step_idx + 1 == len(reverse_timesteps):
            return pred_sample

        next_t = torch.full(
            (state.shape[0],),
            int(reverse_timesteps[step_idx + 1]),
            device=state.device,
            dtype=torch.long,
        )
        alpha_t, sigma_t = alpha_sigma_for_timesteps(scheduler, current_t, state)
        alpha_next, sigma_next = alpha_sigma_for_timesteps(scheduler, next_t, state)
        epsilon_hat = (state - alpha_t * pred_sample) / sigma_t.clamp_min(1e-6)
        state = alpha_next * pred_sample + sigma_next * epsilon_hat
    raise RuntimeError("reverse_timesteps must be non-empty")


def load_negative_bank(config: dict, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    bank_dir = config.get("negative_bank_dir")
    if not bank_dir:
        semantic_dir = config.get("semantic_feature_dir")
        if semantic_dir:
            bank_dir = repo_path(semantic_dir) / "negative_bank"
    if not bank_dir:
        return None, None
    bank_dir = repo_path(bank_dir)
    easy_path = bank_dir / "easy_negative.npy"
    hard_path = bank_dir / "hard_negative.npy"
    if not easy_path.exists() or not hard_path.exists():
        return None, None
    easy = torch.from_numpy(np.load(easy_path).astype(np.int64)).to(device=device)
    hard = torch.from_numpy(np.load(hard_path).astype(np.int64)).to(device=device)
    return easy, hard


def load_topk_hard_bank(config: dict, device: torch.device) -> torch.Tensor | None:
    """Load top-K text-similar seen-class negatives (shape: 60 x K).

    Negatives are restricted to seen classes at bank-generation time,
    so no unseen-label leakage occurs during training.
    """
    if not bool(config.get("topk_hard_neg", False)):
        return None
    bank_dir = config.get("negative_bank_dir")
    if not bank_dir:
        return None
    bank_dir = repo_path(bank_dir)
    topk_path = bank_dir / "topk_hard_negative.npy"
    if not topk_path.exists():
        print(f"[WARN] topk_hard_neg=True but {topk_path} not found, falling back to single hard_bank.")
        return None
    arr = np.load(topk_path).astype(np.int64)   # (num_classes, K)
    return torch.from_numpy(arr).to(device=device)


def fit_low_rank_ridge(
    text_features: np.ndarray,
    visual_targets: np.ndarray,
    rank: int,
    ridge: float,
) -> dict[str, np.ndarray | int | float]:
    """Fit a centred low-rank text-to-visual ridge regression."""
    text = np.asarray(text_features, dtype=np.float64)
    targets = np.asarray(visual_targets, dtype=np.float64)
    if text.ndim != 2 or targets.ndim != 2 or text.shape[0] != targets.shape[0]:
        raise ValueError(
            "text_features and visual_targets must be rank-2 with matching rows, got "
            f"{text.shape} and {targets.shape}"
        )
    if text.shape[0] < 2:
        raise ValueError("Low-rank prototype bridge requires at least two seen classes")

    text_mean = text.mean(axis=0, keepdims=True)
    target_mean = targets.mean(axis=0, keepdims=True)
    text_centered = text - text_mean
    target_centered = targets - target_mean
    _u, _s, vh = np.linalg.svd(text_centered, full_matrices=False)
    effective_rank = max(1, min(int(rank), vh.shape[0], text.shape[0] - 1))
    basis = vh[:effective_rank].T
    projected = text_centered @ basis
    ridge_value = max(float(ridge), 1e-8)
    gram = projected.T @ projected + np.eye(effective_rank) * ridge_value
    coefficients = np.linalg.solve(gram, projected.T @ target_centered)
    return {
        "text_mean": text_mean.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "basis": basis.astype(np.float32),
        "coefficients": coefficients.astype(np.float32),
        "rank": effective_rank,
        "ridge": ridge_value,
    }


def predict_low_rank_ridge(
    model: dict[str, np.ndarray | int | float],
    text_features: np.ndarray,
) -> np.ndarray:
    text = np.asarray(text_features, dtype=np.float32)
    projected = (text - model["text_mean"]) @ model["basis"]
    return (model["target_mean"] + projected @ model["coefficients"]).astype(np.float32)


def select_low_rank_ridge(
    text_features: np.ndarray,
    visual_targets: np.ndarray,
    rank_candidates: list[int],
    ridge_candidates: list[float],
    num_folds: int,
    seed: int,
) -> tuple[int, float, float]:
    """Select bridge capacity using deterministic seen-class cross-validation."""
    text = np.asarray(text_features, dtype=np.float32)
    targets = np.asarray(visual_targets, dtype=np.float32)
    class_count = text.shape[0]
    fold_count = max(2, min(int(num_folds), class_count))
    order = np.random.default_rng(int(seed)).permutation(class_count)
    folds = [fold for fold in np.array_split(order, fold_count) if fold.size]
    ranks = sorted(set(max(1, int(value)) for value in rank_candidates))
    ridges = sorted(set(max(float(value), 1e-8) for value in ridge_candidates))
    if not ranks or not ridges:
        raise ValueError("Prototype bridge rank/ridge candidate lists must be non-empty")

    errors = {(rank, ridge): [] for rank in ranks for ridge in ridges}
    all_indices = np.arange(class_count)
    for validation_indices in folds:
        training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        for rank in ranks:
            for ridge in ridges:
                model = fit_low_rank_ridge(
                    text[training_indices],
                    targets[training_indices],
                    rank,
                    ridge,
                )
                prediction = predict_low_rank_ridge(model, text[validation_indices])
                errors[(rank, ridge)].append(float(np.mean((prediction - targets[validation_indices]) ** 2)))

    best_pair, fold_errors = min(errors.items(), key=lambda item: float(np.mean(item[1])))
    return int(best_pair[0]), float(best_pair[1]), float(np.mean(fold_errors))


def cross_validated_low_rank_predictions(
    text_features: np.ndarray,
    visual_targets: np.ndarray,
    rank: int,
    ridge: float,
    num_folds: int,
    seed: int,
) -> np.ndarray:
    """Predict every seen class from a fold that excluded that class."""
    text = np.asarray(text_features, dtype=np.float32)
    targets = np.asarray(visual_targets, dtype=np.float32)
    class_count = text.shape[0]
    fold_count = max(2, min(int(num_folds), class_count))
    order = np.random.default_rng(int(seed)).permutation(class_count)
    folds = [fold for fold in np.array_split(order, fold_count) if fold.size]
    predictions = np.empty_like(targets)
    all_indices = np.arange(class_count)
    for validation_indices in folds:
        training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        model = fit_low_rank_ridge(
            text[training_indices],
            targets[training_indices],
            rank,
            ridge,
        )
        predictions[validation_indices] = predict_low_rank_ridge(model, text[validation_indices])
    return predictions


def build_seen_bridge_context(
    config: dict,
    unseen_labels: np.ndarray,
    text_embed: torch.Tensor,
    device: torch.device,
) -> dict | None:
    """Precompute a seen-only bridge from skeleton features to unseen classes.

    Generic across ANY seen/unseen split (55/5, 48/12, 40/20, 30/30, ...):
    no unseen-specific hardcoding. The bridge exploits the fact that unseen
    test features still live near SOME seen-class neighbourhood in feature
    space, and that neighbourhood's text embedding tells us something about
    which unseen class is semantically closest.

    Algorithm (fully label-free w.r.t. unseen classes):
      1. Load raw seen-class TRAIN features (the same feature file used to
         train the DiT) and their seen labels.
      2. For a query test feature x, find its K nearest seen-class train
         neighbours (feature-space Euclidean KNN).
      3. Each neighbour "votes" for every unseen class c, weighted by
         (1/distance) * text_similarity(neighbour_class, c).
      4. This produces one (N_test, num_unseen) bridge score matrix that
         evaluate_zsl can optionally blend with the DiT reconstruction score.

    As seen-class count shrinks (48/12 → 30/30 → 12/48), neighbour text
    similarity to any given unseen class gets noisier and less reliable —
    the bridge naturally contributes less useful signal, but this function
    still returns valid output; the CALLER controls how much weight to give
    it via eval_bridge_alpha (0 disables the bridge entirely).
    """
    if not bool(config.get("eval_bridge", False)):
        return None

    train_path = repo_path(config["train_feeder_args"]["path"])
    x_seen = np.load(train_path / "train.npy").astype(np.float32)
    y_seen = np.load(train_path / "train_label.npy").astype(np.int64)
    unseen_set = set(int(v) for v in unseen_labels.tolist())
    seen_mask = ~np.isin(y_seen, list(unseen_set))
    x_seen = x_seen[seen_mask]
    y_seen = y_seen[seen_mask]
    if x_seen.shape[0] == 0:
        print("[WARN] eval_bridge=True but no seen-class training features found; bridge disabled.")
        return None

    bridge_mode = str(config.get("eval_bridge_mode", "knn")).lower()
    if bridge_mode in {"low_rank_prototype", "visual_prototype", "prototype"}:
        flat_seen = x_seen.reshape(x_seen.shape[0], -1)
        feature_mean = flat_seen.mean(axis=0, keepdims=True)
        feature_scale = flat_seen.std(axis=0, keepdims=True)
        feature_scale = np.maximum(feature_scale, float(config.get("eval_bridge_min_scale", 1e-4)))
        standardized = (flat_seen - feature_mean) / feature_scale
        seen_classes = np.asarray(sorted(set(int(v) for v in y_seen.tolist())), dtype=np.int64)
        visual_centroids = np.stack(
            [standardized[y_seen == class_label].mean(axis=0) for class_label in seen_classes],
            axis=0,
        ).astype(np.float32)

        pooled = F.normalize(pooled_text_embed(text_embed).detach().float(), dim=-1)
        pooled_np = pooled.cpu().numpy().astype(np.float32)
        seen_text = pooled_np[seen_classes]
        rank_candidates = config.get("eval_bridge_rank_candidates", [4, 8, 16, 32])
        ridge_candidates = config.get("eval_bridge_ridge_candidates", [0.01, 0.1, 1.0, 10.0])
        selected_rank, selected_ridge, cv_mse = select_low_rank_ridge(
            seen_text,
            visual_centroids,
            [int(value) for value in rank_candidates],
            [float(value) for value in ridge_candidates],
            int(config.get("eval_bridge_cv_folds", 5)),
            int(config.get("seed", 2025)),
        )
        target_variance = float(np.mean((visual_centroids - visual_centroids.mean(axis=0)) ** 2))
        cv_reliability = float(np.clip(1.0 - cv_mse / max(target_variance, 1e-8), 0.0, 1.0))
        regression = fit_low_rank_ridge(
            seen_text,
            visual_centroids,
            selected_rank,
            selected_ridge,
        )
        unseen_prototypes = predict_low_rank_ridge(regression, pooled_np[unseen_labels])
        return {
            "mode": "low_rank_prototype",
            "prototypes": torch.from_numpy(unseen_prototypes).to(device=device),
            "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)).to(device=device),
            "feature_scale": torch.from_numpy(feature_scale.astype(np.float32)).to(device=device),
            "rank": int(regression["rank"]),
            "ridge": float(regression["ridge"]),
            "cv_mse": cv_mse,
            "target_variance": target_variance,
            "cv_reliability": cv_reliability,
            "description": (
                f"low_rank_prototype seen_classes={len(seen_classes)} "
                f"rank={int(regression['rank'])} ridge={float(regression['ridge']):g} "
                f"seen_cv_mse={cv_mse:.6f} reliability={cv_reliability:.3f}"
            ),
        }
    if bridge_mode not in {"knn", "seen_knn"}:
        raise ValueError(f"Unsupported eval_bridge_mode: {bridge_mode}")

    # Text similarity: seen_class -> each unseen_class, computed once.
    pooled = pooled_text_embed(text_embed).detach().float()
    pooled_n = F.normalize(pooled, dim=-1)
    sim_full = (pooled_n @ pooled_n.T).cpu().numpy()   # (num_classes, num_classes) cosine sim

    seen_classes = sorted(set(int(v) for v in y_seen.tolist()))
    seen_to_row = {c: i for i, c in enumerate(seen_classes)}
    text_bridge = np.stack(
        [[sim_full[s, u] for u in unseen_labels.tolist()] for s in seen_classes],
        axis=0,
    ).astype(np.float32)  # (num_seen_classes_present, num_unseen)

    return {
        "mode": "knn",
        "x_seen": torch.from_numpy(x_seen).to(device=device),         # (M, feat_dim)
        "y_seen_row": torch.tensor(
            [seen_to_row[int(c)] for c in y_seen.tolist()], device=device, dtype=torch.long
        ),                                                            # (M,) -> row into text_bridge
        "text_bridge": torch.from_numpy(text_bridge).to(device=device),  # (num_seen_classes_present, num_unseen)
        "k": int(config.get("eval_bridge_k", 10)),
        "description": f"knn seen_samples={x_seen.shape[0]} k={int(config.get('eval_bridge_k', 10))}",
    }


@torch.no_grad()
def compute_bridge_scores(
    features: torch.Tensor,
    bridge_ctx: dict,
) -> torch.Tensor:
    """KNN seen-class bridge score for a batch of test features.

    Returns (B, num_unseen) where LOWER is better (matches reconstruction-
    distance convention, so it can be blended directly with DiT scores).
    """
    feat_flat = features.flatten(1).float()             # (B, D)
    if bridge_ctx.get("mode") == "low_rank_prototype":
        standardized = (feat_flat - bridge_ctx["feature_mean"]) / bridge_ctx["feature_scale"]
        prototypes = bridge_ctx["prototypes"].float()
        return (standardized[:, None, :] - prototypes[None, :, :]).pow(2).mean(dim=-1)

    x_seen = bridge_ctx["x_seen"].flatten(1).float()      # (M, D)
    k = bridge_ctx["k"]

    # Pairwise distance, chunked over the seen pool would be ideal for huge
    # M, but M is at most a few tens of thousands of 256-d vectors here, so a
    # single cdist call is fine.
    dists = torch.cdist(feat_flat, x_seen)               # (B, M)
    topk_dist, topk_idx = torch.topk(dists, k=min(k, x_seen.shape[0]), dim=1, largest=False)
    weights = 1.0 / (topk_dist + 1e-6)                    # (B, k)

    neighbour_rows = bridge_ctx["y_seen_row"][topk_idx]   # (B, k) -> row into text_bridge
    neighbour_text = bridge_ctx["text_bridge"][neighbour_rows]  # (B, k, num_unseen)

    # Weighted vote: higher text similarity + closer neighbour => more
    # confident this unseen class is correct => LOWER score (we negate).
    weighted = (weights.unsqueeze(-1) * neighbour_text).sum(dim=1)  # (B, num_unseen)
    weighted = weighted / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -weighted   # lower = more likely, matches reconstruction distance sign


def build_prototype_direction_context(
    config: dict,
    text_embed: torch.Tensor,
    seen_labels: np.ndarray,
    device: torch.device,
) -> dict:
    """Build seen-only visual directions and cross-modal pair reliability."""
    train_path = repo_path(config["train_feeder_args"]["path"])
    features = np.load(train_path / "train.npy").astype(np.float32)
    labels = np.load(train_path / "train_label.npy").astype(np.int64)
    seen = np.asarray(sorted(int(value) for value in seen_labels.tolist()), dtype=np.int64)
    mask = np.isin(labels, seen)
    flat_features = features[mask].reshape(int(mask.sum()), -1)
    labels = labels[mask]
    feature_mean = flat_features.mean(axis=0, keepdims=True)
    feature_scale = np.maximum(
        flat_features.std(axis=0, keepdims=True),
        float(
            config.get(
                "prototype_geometry_min_scale",
                config.get("prototype_direction_min_scale", 1e-4),
            )
        ),
    )
    standardized = (flat_features - feature_mean) / feature_scale
    centroids = np.stack(
        [standardized[labels == class_label].mean(axis=0) for class_label in seen],
        axis=0,
    ).astype(np.float32)

    pooled = F.normalize(pooled_text_embed(text_embed).detach().float(), dim=-1)
    pooled_np = pooled.cpu().numpy().astype(np.float32)
    seen_text = pooled_np[seen]
    rank_candidates = config.get(
        "prototype_geometry_rank_candidates",
        config.get("prototype_direction_rank_candidates", [4, 8, 16, 32]),
    )
    ridge_candidates = config.get(
        "prototype_geometry_ridge_candidates",
        config.get("prototype_direction_ridge_candidates", [0.01, 0.1, 1.0, 10.0]),
    )
    cv_folds = int(
        config.get(
            "prototype_geometry_cv_folds",
            config.get("prototype_direction_cv_folds", 5),
        )
    )
    selected_rank, selected_ridge, cv_mse = select_low_rank_ridge(
        seen_text,
        centroids,
        [int(value) for value in rank_candidates],
        [float(value) for value in ridge_candidates],
        cv_folds,
        int(config.get("seed", 2025)),
    )
    predicted_centroids = cross_validated_low_rank_predictions(
        seen_text,
        centroids,
        selected_rank,
        selected_ridge,
        cv_folds,
        int(config.get("seed", 2025)),
    )

    actual_direction = centroids[:, None, :] - centroids[None, :, :]
    predicted_direction = predicted_centroids[:, None, :] - predicted_centroids[None, :, :]
    actual_norm = np.linalg.norm(actual_direction, axis=-1)
    predicted_norm = np.linalg.norm(predicted_direction, axis=-1)
    denominator = np.maximum(actual_norm * predicted_norm, 1e-8)
    pair_reliability = np.sum(actual_direction * predicted_direction, axis=-1) / denominator
    pair_reliability = np.clip(pair_reliability, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(pair_reliability, 0.0)

    off_diagonal = ~np.eye(len(seen), dtype=bool)
    sorted_distances = np.sort(actual_norm[off_diagonal])
    if sorted_distances.size < 2:
        distance_percentile = np.full_like(actual_norm, 0.5, dtype=np.float32)
    else:
        left_rank = np.searchsorted(sorted_distances, actual_norm, side="left")
        right_rank = np.searchsorted(sorted_distances, actual_norm, side="right")
        average_rank = 0.5 * (left_rank + right_rank - 1)
        distance_percentile = np.clip(
            average_rank / float(sorted_distances.size - 1),
            0.0,
            1.0,
        ).astype(np.float32)
    np.fill_diagonal(distance_percentile, 0.5)

    num_classes = int(text_embed.shape[0])
    centroid_bank = np.zeros((num_classes, centroids.shape[1]), dtype=np.float32)
    reliability_bank = np.zeros((num_classes, num_classes), dtype=np.float32)
    distance_percentile_bank = np.full((num_classes, num_classes), 0.5, dtype=np.float32)
    centroid_bank[seen] = centroids
    reliability_bank[np.ix_(seen, seen)] = pair_reliability
    distance_percentile_bank[np.ix_(seen, seen)] = distance_percentile
    nonzero = pair_reliability[pair_reliability > 0]
    mean_reliability = float(nonzero.mean()) if nonzero.size else 0.0
    return {
        "centroids": torch.from_numpy(centroid_bank).to(device=device),
        "feature_scale": torch.from_numpy(feature_scale.astype(np.float32)).to(device=device),
        "pair_reliability": torch.from_numpy(reliability_bank).to(device=device),
        "distance_percentile": torch.from_numpy(distance_percentile_bank).to(device=device),
        "description": (
            f"seen_classes={len(seen)} rank={selected_rank} "
            f"ridge={selected_ridge:g} seen_cv_mse={cv_mse:.6f} "
            f"mean_pair_reliability={mean_reliability:.3f}"
        ),
    }


def prototype_direction_contrastive_loss(
    positive_sample: torch.Tensor,
    negative_sample: torch.Tensor,
    labels: torch.Tensor,
    negative_labels: torch.Tensor,
    context: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align the text-conditioned counterfactual shift with visual class geometry."""
    predicted_direction = (negative_sample - positive_sample).flatten(1).float()
    predicted_direction = predicted_direction / context["feature_scale"]
    visual_direction = context["centroids"][negative_labels] - context["centroids"][labels]
    cosine = F.cosine_similarity(predicted_direction, visual_direction.float(), dim=1, eps=1e-6)
    reliability = context["pair_reliability"][labels, negative_labels].float()
    valid = labels.ne(negative_labels).float()
    loss = ((1.0 - cosine) * reliability * valid).mean()
    mean_reliability = (reliability * valid).sum() / valid.sum().clamp_min(1.0)
    return loss, mean_reliability


def reliability_gated_adaptive_margin(
    labels: torch.Tensor,
    negative_labels: torch.Tensor,
    context: dict,
    base_margin: float,
    min_margin: float,
    max_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend visual-distance margins with the A0 margin using OOF reliability."""
    if not 0.0 <= min_margin <= base_margin <= max_margin:
        raise ValueError(
            "Adaptive margins must satisfy 0 <= min_margin <= base_margin <= max_margin"
        )
    reliability = context["pair_reliability"][labels, negative_labels].float()
    distance_percentile = context["distance_percentile"][labels, negative_labels].float()
    visual_margin = min_margin + (max_margin - min_margin) * distance_percentile
    margin = base_margin + reliability * (visual_margin - base_margin)
    valid = labels.ne(negative_labels).float()
    valid_count = valid.sum().clamp_min(1.0)
    mean_margin = (margin * valid).sum() / valid_count
    mean_reliability = (reliability * valid).sum() / valid_count
    return margin, mean_margin, mean_reliability


def noise_consistent_energy_ranking_loss(
    positive_distances: torch.Tensor,
    negative_distances: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank the lower confidence bound of the energy gap across noise views."""
    if positive_distances.ndim != 2 or positive_distances.shape != negative_distances.shape:
        raise ValueError(
            "Noise-consistent distances must have matching (views, batch) shapes, got "
            f"{tuple(positive_distances.shape)} and {tuple(negative_distances.shape)}"
        )
    if positive_distances.shape[0] < 2:
        raise ValueError("Noise-consistent ranking requires at least two noise views")
    if beta < 0.0:
        raise ValueError("noise_consistent_beta must be non-negative")

    gap = negative_distances - positive_distances
    gap_mean_per_sample = gap.mean(dim=0)
    gap_std_per_sample = gap.std(dim=0, unbiased=False)
    rank_violation = float(margin) - gap_mean_per_sample + float(beta) * gap_std_per_sample
    valid = valid_mask.to(dtype=rank_violation.dtype)
    valid_count = valid.sum().clamp_min(1.0)
    ranking_loss = torch.clamp(rank_violation, min=0.0) * valid
    mean_gap = (gap_mean_per_sample * valid).sum() / valid_count
    mean_gap_std = (gap_std_per_sample * valid).sum() / valid_count
    active_fraction = (
        rank_violation.gt(0).to(dtype=valid.dtype).mul(valid).sum() / valid_count
    )
    return ranking_loss, mean_gap, mean_gap_std, active_fraction


def loss_aware_timestep_probabilities(
    second_moment: torch.Tensor,
    uniform_mix: float,
) -> torch.Tensor:
    """Loss-second-moment schedule sampler with a nonzero uniform floor."""
    if second_moment.ndim != 1 or second_moment.numel() == 0:
        raise ValueError("Loss-aware timestep moments must be a non-empty vector")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("loss_aware_uniform_mix must be in [0, 1]")
    weights = second_moment.detach().clamp_min(1e-12).sqrt()
    normalized = weights / weights.sum().clamp_min(1e-12)
    uniform = torch.full_like(normalized, 1.0 / normalized.numel())
    return float(uniform_mix) * uniform + (1.0 - float(uniform_mix)) * normalized


def scale_calibrated_timestep_gap_loss(
    anchor_gap: torch.Tensor,
    partner_gap: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_margin: float,
    partner_floor: float,
    huber_delta: float,
    max_anchor_gap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match pairwise energy gaps across timesteps after batch scale calibration.

    Conditional reconstruction energies at distinct diffusion timesteps are not
    repeated measurements of a shared scalar: their gap scales differ.  The
    t=25 anchor is therefore detached and both gaps are normalized by their
    reliable-batch means before Huber matching.  A small hinge prevents the
    partner timestep from preserving a scaled but negative class ordering.
    """
    if anchor_gap.shape != partner_gap.shape or anchor_gap.ndim != 1:
        raise ValueError("Timestep gap consistency expects matching (batch,) gaps")
    if anchor_margin < 0.0 or partner_floor < 0.0 or huber_delta <= 0.0:
        raise ValueError("Invalid timestep gap consistency hyperparameters")

    valid = valid_mask.bool()
    reliable = valid & anchor_gap.detach().gt(float(anchor_margin))
    if max_anchor_gap is not None and math.isfinite(float(max_anchor_gap)):
        reliable = reliable & anchor_gap.detach().le(float(max_anchor_gap))
    coverage = reliable.float().mean()
    if not bool(reliable.any()):
        zero = anchor_gap.new_zeros(())
        return zero, coverage, zero, zero

    anchor_values = anchor_gap.detach()[reliable]
    partner_values = partner_gap[reliable]
    anchor_scale = anchor_values.mean().clamp_min(1e-6)
    partner_scale = partner_values.detach().abs().mean().clamp_min(1e-6)
    residual = partner_values / partner_scale - anchor_values / anchor_scale
    absolute = residual.abs()
    delta = float(huber_delta)
    consistency = torch.where(
        absolute <= delta,
        0.5 * residual.square() / delta,
        absolute - 0.5 * delta,
    ).mean()
    ordering = F.relu(float(partner_floor) - partner_values).mean()
    return consistency + ordering, coverage, anchor_values.mean(), partner_values.mean()


def select_teacher_guided_timesteps(
    teacher_gaps: torch.Tensor,
    candidate_timesteps: torch.Tensor,
    margin: float,
    temperature: float,
    anchor_timestep: int,
    anchor_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample valid semi-hard timesteps, with a fixed anchor fallback."""
    if teacher_gaps.ndim != 2:
        raise ValueError(
            f"teacher_gaps must have shape (candidates, batch), got {tuple(teacher_gaps.shape)}"
        )
    if candidate_timesteps.ndim != 1 or candidate_timesteps.numel() != teacher_gaps.shape[0]:
        raise ValueError("candidate_timesteps must match the first teacher_gaps dimension")
    if not 0.0 <= anchor_probability <= 1.0:
        raise ValueError("adaptive_timestep_anchor_probability must be in [0, 1]")
    tau = max(float(temperature), 1e-6)
    candidates = candidate_timesteps.to(device=teacher_gaps.device, dtype=torch.long)
    anchor_index = torch.argmin((candidates - int(anchor_timestep)).abs())

    valid = teacher_gaps.gt(0.0)
    logits = -(teacher_gaps - float(margin)).abs() / tau
    logits = logits.masked_fill(~valid, -torch.inf)
    all_invalid = ~valid.any(dim=0)
    if bool(all_invalid.any()):
        logits[:, all_invalid] = -torch.inf
        logits[anchor_index, all_invalid] = 0.0
    probabilities = torch.softmax(logits, dim=0)
    selected = torch.multinomial(probabilities.transpose(0, 1), 1).squeeze(1)

    use_anchor = torch.rand(teacher_gaps.shape[1], device=teacher_gaps.device).lt(
        float(anchor_probability)
    )
    selected = torch.where(use_anchor, anchor_index.expand_as(selected), selected)
    return selected, probabilities


def build_seen_analogy_context(
    config: dict,
    unseen_labels: np.ndarray,
    text_embed: torch.Tensor,
    device: torch.device,
) -> dict | None:
    """Precompute seen "role-model" centroids for each unseen class (plan:
    borrow a discriminative direction from an analogous, ALREADY-SEPARABLE
    seen pair).

    Motivation (measured on the 55/5 split): reading (10) and writing (11)
    are confusable partly because every per-candidate "most similar seen
    classes" selection overlaps almost completely (7/8 shared in earlier
    AnchorDiT experiments) -- there is no unique discriminative evidence
    to pool from EACH candidate's own neighbourhood independently.

    Instead: assign every unseen class a UNIQUE seen "role model" via
    bipartite matching on text similarity (Hungarian algorithm), so no two
    unseen classes share the same seen proxy. Unlike a per-candidate top-k
    lookup, uniqueness is enforced globally, which is what makes the
    resulting seen role-model CENTROIDS mutually separable in feature space
    (they are real, distinct, labelled seen classes -- verified empirically:
    projecting unseen test features onto the seen-pair difference direction
    mu_a - mu_b for the (10,11) role-model pair gives separation 0.765,
    vs 0.291 average / 0.798 max over 20 random seen pairs).

    Generic across ANY seen/unseen split size: bipartite matching handles
    any num_unseen <= num_seen (55/5, 48/12, 40/20, ...); if num_unseen >
    num_seen (e.g. some 30/30 or larger-unseen splits), falls back to
    greedy per-class argmax with re-use allowed rather than failing.
    """
    if not bool(config.get("eval_seen_analogy", False)):
        return None

    train_path = repo_path(config["train_feeder_args"]["path"])
    x_seen = np.load(train_path / "train.npy").astype(np.float32)
    y_seen = np.load(train_path / "train_label.npy").astype(np.int64)
    unseen_set = set(int(v) for v in unseen_labels.tolist())
    seen_classes = sorted(set(int(v) for v in y_seen.tolist()) - unseen_set)
    if not seen_classes:
        print("[WARN] eval_seen_analogy=True but no seen classes found; disabled.")
        return None

    pooled = pooled_text_embed(text_embed).detach().float()
    pooled_n = F.normalize(pooled, dim=-1)
    sim = (pooled_n @ pooled_n.T).cpu().numpy()   # (num_classes, num_classes)

    unseen_list = [int(v) for v in unseen_labels.tolist()]
    if len(unseen_list) <= len(seen_classes):
        try:
            from scipy.optimize import linear_sum_assignment
            cost = np.array([[-sim[u, s] for s in seen_classes] for u in unseen_list])
            row_idx, col_idx = linear_sum_assignment(cost)
            role_model = {unseen_list[r]: seen_classes[c] for r, c in zip(row_idx, col_idx)}
        except ImportError:
            print("[WARN] scipy unavailable, falling back to greedy (non-unique) role-model assignment.")
            role_model = {u: max(seen_classes, key=lambda s: sim[u, s]) for u in unseen_list}
    else:
        # More unseen classes than seen classes: uniqueness impossible, fall
        # back to greedy per-class argmax (role models may repeat).
        role_model = {u: max(seen_classes, key=lambda s: sim[u, s]) for u in unseen_list}

    centroids = []
    for u in unseen_list:
        s = role_model[u]
        cls_feats = x_seen[y_seen == s]
        centroids.append(cls_feats.reshape(cls_feats.shape[0], -1).mean(axis=0))
    centroid_mat = np.stack(centroids, axis=0).astype(np.float32)   # (num_unseen, D)

    print(
        "Seen-analogy role models: "
        + ", ".join(f"{u}<-{role_model[u]}(sim={sim[u, role_model[u]]:.3f})" for u in unseen_list)
    )

    return {
        "centroids": torch.from_numpy(centroid_mat).to(device=device),   # (num_unseen, D)
        "role_model": role_model,
    }


@torch.no_grad()
def compute_seen_analogy_scores(
    features: torch.Tensor,
    analogy_ctx: dict,
) -> torch.Tensor:
    """Distance from each test feature to each unseen class's seen role-model
    centroid. Returns (B, num_unseen), lower = more likely (same convention
    as reconstruction_distance / compute_bridge_scores).
    """
    feat_flat = features.flatten(1).float()                # (B, D)
    centroids = analogy_ctx["centroids"].flatten(1).float() # (num_unseen, D)
    return torch.cdist(feat_flat, centroids)                # (B, num_unseen)


def reconstruction_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))


def sample_distinct_negative_labels(
    labels: torch.Tensor,
    seen_labels: list[int],
    excluded_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one seen negative per row, excluding the positive and optional hard negative."""
    positives = [int(item) for item in labels.detach().cpu().tolist()]
    excluded = (
        [int(item) for item in excluded_labels.detach().cpu().tolist()]
        if excluded_labels is not None
        else [None] * len(positives)
    )
    negatives: list[int] = []
    for positive, extra_excluded in zip(positives, excluded):
        pool = [
            int(candidate)
            for candidate in seen_labels
            if int(candidate) != positive and int(candidate) != extra_excluded
        ]
        if not pool:
            pool = [int(candidate) for candidate in seen_labels if int(candidate) != positive]
        if not pool:
            raise ValueError("At least two seen classes are required for contrastive training.")
        negatives.append(random.choice(pool))
    return torch.tensor(negatives, device=labels.device, dtype=torch.long)


def build_ambiguity_neighbor_bank(
    config: dict,
    text_embed: torch.Tensor,
    seen_labels: np.ndarray | list[int],
) -> tuple[dict[int, list[int]], str]:
    """Find conservative seen-only neighbours shared by text and visual geometry."""
    seen = np.asarray(sorted(int(value) for value in seen_labels), dtype=np.int64)
    if seen.size < 2:
        raise ValueError("Ambiguity-aware contrast requires at least two seen classes")

    train_path = repo_path(config["train_feeder_args"]["path"])
    features = np.load(train_path / "train.npy").astype(np.float32).reshape(-1, int(config.get("in_channels", 256)))
    labels = np.load(train_path / "train_label.npy").astype(np.int64)
    sample_norm = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    normalized_features = features / sample_norm
    visual_prototypes = np.stack(
        [normalized_features[labels == class_label].mean(axis=0) for class_label in seen],
        axis=0,
    )
    prototype_norm = np.maximum(np.linalg.norm(visual_prototypes, axis=1, keepdims=True), 1e-8)
    visual_prototypes = visual_prototypes / prototype_norm

    text_prototypes = pooled_text_embed(text_embed).detach().float().cpu().numpy()[seen]
    text_norm = np.maximum(np.linalg.norm(text_prototypes, axis=1, keepdims=True), 1e-8)
    text_prototypes = text_prototypes / text_norm
    text_similarity = text_prototypes @ text_prototypes.T
    visual_similarity = visual_prototypes @ visual_prototypes.T
    np.fill_diagonal(text_similarity, -np.inf)
    np.fill_diagonal(visual_similarity, -np.inf)

    text_topk = min(max(1, int(config.get("ambiguity_text_topk", 3))), seen.size - 1)
    visual_topk = min(max(1, int(config.get("ambiguity_visual_topk", 3))), seen.size - 1)
    text_positions = np.argpartition(-text_similarity, text_topk - 1, axis=1)[:, :text_topk]
    visual_positions = np.argpartition(-visual_similarity, visual_topk - 1, axis=1)[:, :visual_topk]
    text_mask = np.zeros((seen.size, seen.size), dtype=bool)
    visual_mask = np.zeros_like(text_mask)
    rows = np.arange(seen.size)[:, None]
    text_mask[rows, text_positions] = True
    visual_mask[rows, visual_positions] = True
    ambiguity_mask = text_mask & visual_mask
    if bool(config.get("ambiguity_symmetrize", True)):
        ambiguity_mask |= ambiguity_mask.T
    np.fill_diagonal(ambiguity_mask, False)

    bank = {
        int(seen[row]): [int(seen[col]) for col in np.flatnonzero(ambiguity_mask[row])]
        for row in range(seen.size)
    }
    counts = np.asarray([len(bank[int(label)]) for label in seen], dtype=np.int64)
    description = (
        f"seen_classes={seen.size} text_topk={text_topk} visual_topk={visual_topk} "
        f"symmetrize={bool(config.get('ambiguity_symmetrize', True))} "
        f"mean_neighbors={counts.mean():.3f} covered_classes={int((counts > 0).sum())}/{seen.size}"
    )
    return bank, description


def sample_ambiguity_aware_negative_labels(
    labels: torch.Tensor,
    seen_labels: list[int],
    ambiguity_bank: dict[int, list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one safe random negative and one potentially ambiguous negative."""
    positives = [int(item) for item in labels.detach().cpu().tolist()]
    seen_pool = [int(item) for item in seen_labels]
    safe_labels: list[int] = []
    ambiguous_labels: list[int] = []
    ambiguous_valid: list[bool] = []
    for positive in positives:
        ambiguous_pool = [
            candidate
            for candidate in ambiguity_bank.get(positive, [])
            if candidate != positive
        ]
        ambiguous_set = set(ambiguous_pool)
        safe_pool = [
            candidate
            for candidate in seen_pool
            if candidate != positive and candidate not in ambiguous_set
        ]
        if not safe_pool:
            safe_pool = [candidate for candidate in seen_pool if candidate != positive]
        if not safe_pool:
            raise ValueError("Ambiguity-aware contrast requires at least two seen classes")
        safe_label = int(random.choice(safe_pool))
        safe_labels.append(safe_label)
        if ambiguous_pool:
            ambiguous_labels.append(int(random.choice(ambiguous_pool)))
            ambiguous_valid.append(True)
        else:
            ambiguous_labels.append(safe_label)
            ambiguous_valid.append(False)
    device = labels.device
    return (
        torch.tensor(safe_labels, device=device, dtype=torch.long),
        torch.tensor(ambiguous_labels, device=device, dtype=torch.long),
        torch.tensor(ambiguous_valid, device=device, dtype=torch.bool),
    )


def build_semantic_neighbor_bank(
    text_features: torch.Tensor,
    seen_labels: np.ndarray | list[int],
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build seen-only text neighbours from the exact conditioning features in use."""
    normalized_text = F.normalize(text_features.detach().float(), dim=-1)
    seen = torch.as_tensor(seen_labels, device=normalized_text.device, dtype=torch.long)
    k = int(num_neighbors)
    if k < 1:
        raise ValueError(f"semantic_topk must be positive, got {k}")
    if seen.numel() <= k:
        raise ValueError(
            f"semantic_topk={k} requires at least {k + 1} seen classes, got {seen.numel()}"
        )

    seen_similarity = normalized_text[seen] @ normalized_text[seen].T
    seen_similarity.fill_diagonal_(-torch.inf)
    neighbor_positions = torch.topk(seen_similarity, k=k, dim=1).indices
    neighbors = seen[neighbor_positions]
    bank = torch.full(
        (normalized_text.shape[0], k),
        -1,
        device=normalized_text.device,
        dtype=torch.long,
    )
    bank[seen] = neighbors
    return bank, normalized_text


def semantic_topology_distillation_loss(
    distance_matrix: torch.Tensor,
    candidate_labels: torch.Tensor,
    anchor_labels: torch.Tensor,
    normalized_text: torch.Tensor,
    teacher_tau: float,
    student_tau: float,
) -> torch.Tensor:
    """Match reconstruction-score topology to the conditioning-text topology."""
    if distance_matrix.shape != candidate_labels.shape:
        raise ValueError(
            "distance_matrix and candidate_labels must have identical shapes, got "
            f"{tuple(distance_matrix.shape)} and {tuple(candidate_labels.shape)}"
        )
    teacher_tau = max(float(teacher_tau), 1e-6)
    student_tau = max(float(student_tau), 1e-6)

    anchor_text = normalized_text[anchor_labels]
    candidate_text = normalized_text[candidate_labels]
    text_similarity = torch.einsum("bd,bcd->bc", anchor_text, candidate_text)

    # A0's random draw can rarely equal the positive or a semantic neighbour.
    # Keep its triplet behaviour unchanged, but count each class only once in KD.
    valid = torch.ones_like(candidate_labels, dtype=torch.bool)
    for col in range(1, candidate_labels.shape[1]):
        valid[:, col] = ~candidate_labels[:, col, None].eq(candidate_labels[:, :col]).any(dim=1)

    masked_value = -1e4
    teacher_logits = (text_similarity / teacher_tau).masked_fill(~valid, masked_value)
    student_logits = (-distance_matrix.float() / student_tau).masked_fill(~valid, masked_value)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
    student_log_probs = F.log_softmax(student_logits, dim=1)

    # Scaling by tau keeps gradients in reconstruction-distance units.
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * student_tau


def semantic_distill_curriculum_weight(
    step: int,
    peak_weight: float,
    warmup_steps: int,
    decay_end_steps: int = 0,
) -> float:
    """Warm up semantic KD, then optionally decay it to zero for A0 fine-tuning."""
    step = max(1, int(step))
    peak_weight = max(0.0, float(peak_weight))
    warmup_steps = max(0, int(warmup_steps))
    decay_end_steps = max(0, int(decay_end_steps))

    if warmup_steps > 0 and step < warmup_steps:
        return peak_weight * float(step) / float(warmup_steps)
    if decay_end_steps > warmup_steps:
        if step >= decay_end_steps:
            return 0.0
        return peak_weight * float(decay_end_steps - step) / float(
            decay_end_steps - warmup_steps
        )
    return peak_weight


def generated_contrast_loss(
    aux: C2UAuxHeads,
    anchor_feature: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_easy: torch.Tensor,
    pred_hard: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    anchor = aux.project_feature(anchor_feature)
    pos = aux.project_feature(pred_pos.flatten(1))
    easy = aux.project_feature(pred_easy.flatten(1))
    hard = aux.project_feature(pred_hard.flatten(1))
    logits = torch.stack(
        [
            torch.sum(anchor * pos, dim=1),
            torch.sum(anchor * easy, dim=1),
            torch.sum(anchor * hard, dim=1),
        ],
        dim=1,
    ) / tau
    targets = torch.zeros((logits.shape[0],), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


def sample_infonce_negative_labels(
    labels: torch.Tensor,
    seen_labels: list[int],
    num_neg: int,
    device: torch.device,
    hard_bank: torch.Tensor | None = None,
    topk_hard_bank: torch.Tensor | None = None,
    force_hard_pairs: dict[int, list[int]] | None = None,
) -> torch.Tensor:
    """Sample InfoNCE negative labels for a batch.

    Priority order for filling the negative slots:
      1. top-K text-similar seen-class negatives (topk_hard_bank, slots 0..K-1)
      2. single hard negative from hard_bank (if topk_hard_bank absent)
      3. force_hard_pairs entries (yaml-configured overrides)
      4. random seen-class labels to fill remaining slots

    topk_hard_bank contains only SEEN class indices (generated with unseen
    labels excluded), so no unseen data leaks into training negatives.
    """
    bsz = labels.shape[0]
    label_values = labels.detach().cpu().tolist()
    pool = [int(item) for item in seen_labels]
    if not pool:
        pool = list(range(int(labels.max().detach().cpu().item()) + 1))
    rows: list[list[int]] = []
    force_hard_pairs = force_hard_pairs or {}

    topk_cpu: list[list[int]] | None = None
    if topk_hard_bank is not None:
        topk_cpu = topk_hard_bank.detach().cpu().tolist()

    for row_idx, label in enumerate(label_values):
        y = int(label)
        chosen: list[int] = []
        if topk_cpu is not None:
            for h in topk_cpu[y]:
                h = int(h)
                if h != y and h not in chosen:
                    chosen.append(h)
        elif hard_bank is not None:
            hard = int(hard_bank[int(y)].detach().cpu().item())
            if hard != y:
                chosen.append(hard)
        for forced in force_hard_pairs.get(y, []):
            forced = int(forced)
            if forced != y and forced not in chosen:
                chosen.append(forced)
        random_pool = [item for item in pool if item != y and item not in chosen]
        random.shuffle(random_pool)
        chosen.extend(random_pool[: max(0, int(num_neg) - len(chosen))])
        while len(chosen) < int(num_neg):
            fallback = random.choice(pool)
            if fallback != y:
                chosen.append(int(fallback))
        rows.append(chosen[: int(num_neg)])
    return torch.tensor(rows, device=device, dtype=torch.long)


def reconstruction_infonce_loss(
    model: DiT,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    negative_labels: torch.Tensor,
    target: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
    tau: float,
    negative_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE over reconstruction distances with full gradient on all candidates.

    Uses gradient checkpointing to avoid OOM when num_neg is large (>7).
    Each candidate forward pass runs with gradient but without storing
    intermediate activations, trading compute for memory.
    """
    candidates = torch.cat([labels[:, None], negative_labels], dim=1)  # (B, 1+K)
    distances: list[torch.Tensor] = []
    use_ckpt = candidates.shape[1] > 8  # gradient checkpointing for large neg sets

    def _forward_distance(fc_, fl_, noisy_, timesteps_, target_):
        pred_ = model(noisy_, timesteps_, fc_, fl_)
        ps_ = prediction_to_sample(pred_, noisy_, timesteps_.long(), scheduler, prediction_type)
        return reconstruction_distance(ps_, target_)

    for col in range(candidates.shape[1]):
        cand_labels = candidates[:, col]
        fc, fl = split_text_condition(text_embed, cand_labels)
        if use_ckpt:
            import torch.utils.checkpoint as ckpt_utils
            dist = ckpt_utils.checkpoint(
                _forward_distance, fc, fl, noisy, timesteps, target,
                use_reentrant=False,
            )
        else:
            dist = _forward_distance(fc, fl, noisy, timesteps, target)
        distances.append(dist)

    distance_matrix = torch.stack(distances, dim=1)            # (B, 1+K) all with grad
    logits = -distance_matrix / float(tau)
    if negative_weights is not None:
        if negative_weights.shape != negative_labels.shape:
            raise ValueError("negative_weights must match negative_labels")
        candidate_weights = torch.cat(
            (torch.ones((labels.shape[0], 1), device=logits.device, dtype=logits.dtype), negative_weights),
            dim=1,
        )
        logits = logits + candidate_weights.clamp_min(1e-6).log()
    targets = torch.zeros((labels.shape[0],), device=labels.device, dtype=torch.long)
    return F.cross_entropy(logits, targets), distance_matrix


def episodic_gallery_energy_ranking_loss(
    model: DiT,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    target: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
    gallery_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank queries against one class-disjoint pseudo-unseen gallery.

    The gallery is shared by the batch and contains distinct seen classes.
    Only rows belonging to the sampled gallery receive this auxiliary loss.
    This matches ZSL's all-candidate decision more closely than an independent
    random negative while leaving the base reconstruction/triplet objective
    unchanged for every training row.
    """
    candidates = sorted({int(value) for value in seen_labels})
    if len(candidates) < 2:
        raise ValueError("Episodic gallery ranking requires at least two seen classes")
    size = min(max(2, int(gallery_size)), len(candidates))
    anchor = int(labels[torch.randint(labels.shape[0], (1,), device=labels.device)].item())
    pool = [value for value in candidates if value != anchor]
    gallery_values = [anchor] + random.sample(pool, k=size - 1)
    gallery = torch.tensor(gallery_values, device=labels.device, dtype=torch.long)
    matches = labels[:, None].eq(gallery[None, :])
    query_mask = matches.any(dim=1)
    coverage = query_mask.float().mean()
    if not bool(query_mask.any()):
        zero = noisy.new_zeros(())
        return zero, coverage, zero

    energy_views: list[torch.Tensor] = []
    for candidate in gallery_values:
        candidate_labels = torch.full_like(labels, int(candidate))
        candidate_fc, candidate_fl = split_text_condition(text_embed, candidate_labels)
        candidate_pred = model(noisy, timesteps.to(dtype=noisy.dtype), candidate_fc, candidate_fl)
        candidate_sample = prediction_to_sample(
            candidate_pred,
            noisy,
            timesteps.long(),
            scheduler,
            prediction_type,
        )
        energy_views.append(reconstruction_distance(candidate_sample, target))
    energies = torch.stack(energy_views, dim=1)
    targets = matches[query_mask].float().argmax(dim=1)
    tau = max(float(temperature), 1e-6)
    ranking = F.cross_entropy(-energies[query_mask] / tau, targets) * tau
    sorted_energy = energies[query_mask].sort(dim=1).values
    margin = (sorted_energy[:, 1] - sorted_energy[:, 0]).mean()
    return ranking, coverage, margin


def conditional_reconstruction_energies(
    model: DiT,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    text_embed: torch.Tensor,
    candidate_labels: torch.Tensor,
    target: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
) -> torch.Tensor:
    """Return conditional x0 reconstruction energies for a batch of class candidates."""
    if candidate_labels.ndim != 2:
        raise ValueError("candidate_labels must have shape (batch, candidates)")
    energies: list[torch.Tensor] = []
    for column in range(candidate_labels.shape[1]):
        fc, fl = split_text_condition(text_embed, candidate_labels[:, column])
        pred = model(noisy, timesteps.to(dtype=noisy.dtype), fc, fl)
        pred_sample = prediction_to_sample(
            pred,
            noisy,
            timesteps.long(),
            scheduler,
            prediction_type,
        )
        energies.append(reconstruction_distance(pred_sample, target))
    return torch.stack(energies, dim=1)


def row_relative_energies(energies: torch.Tensor) -> torch.Tensor:
    """Remove row offsets and scales so cross-timestep energy rankings are comparable."""
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError("energies must have shape (batch, at least two candidates)")
    centered = energies - energies.mean(dim=1, keepdim=True)
    # Clamp before rsqrt; sqrt(0).clamp_min(eps) can still backpropagate an
    # infinite derivative when all candidate energies temporarily coincide.
    scale = centered.square().mean(dim=1, keepdim=True).clamp_min(1e-8).rsqrt()
    return centered / scale


def multitime_energy_distillation_loss(
    student: DiT,
    teacher: DiT,
    features: torch.Tensor,
    text_embed: torch.Tensor,
    candidate_labels: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
    anchor_timestep: int,
    teacher_timesteps: list[int],
    teacher_weights: list[float],
    teacher_temperature: float,
    student_temperature: float,
    agreement_gate: bool,
    gap_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distill multi-timestep conditional ranking into one physical timestep.

    The teacher never receives unseen labels: candidates are supplied by the
    caller from the seen class set.  A shared Gaussian draw makes differences
    between teacher views attributable to timestep rather than Monte-Carlo
    noise.  High-noise views can only refine an anchor ranking when their
    top-1 candidate agrees with it.
    """
    if not teacher_timesteps or len(teacher_timesteps) != len(teacher_weights):
        raise ValueError("teacher timesteps and weights must be non-empty and aligned")
    if int(anchor_timestep) not in teacher_timesteps:
        raise ValueError("energy-distill teacher timesteps must include the anchor timestep")
    if any(int(timestep) < 0 or int(timestep) >= int(scheduler.config.num_train_timesteps) for timestep in teacher_timesteps):
        raise ValueError("energy-distill teacher timestep is outside the diffusion schedule")
    if any(float(weight) < 0.0 for weight in teacher_weights) or sum(teacher_weights) <= 0.0:
        raise ValueError("energy-distill teacher weights must be non-negative with positive sum")

    noise = torch.randn_like(features)
    teacher_views: list[torch.Tensor] = []
    with torch.no_grad():
        for timestep in teacher_timesteps:
            t_batch = torch.full(
                (features.shape[0],), int(timestep), device=features.device, dtype=torch.long
            )
            noisy = scheduler.add_noise(features, noise, t_batch)
            energies = conditional_reconstruction_energies(
                teacher,
                noisy,
                t_batch,
                text_embed,
                candidate_labels,
                features,
                scheduler,
                prediction_type,
            )
            teacher_views.append(row_relative_energies(energies))

    anchor_index = teacher_timesteps.index(int(anchor_timestep))
    anchor_view = teacher_views[anchor_index]
    weights = torch.as_tensor(teacher_weights, device=features.device, dtype=features.dtype)
    weights = weights / weights.sum()
    fused_teacher = torch.stack(teacher_views, dim=1).mul(weights.view(1, -1, 1)).sum(dim=1)
    if agreement_gate and len(teacher_views) > 1:
        anchor_choice = anchor_view.argmin(dim=1, keepdim=True)
        agreement = torch.stack(
            [view.argmin(dim=1).eq(anchor_choice.squeeze(1)).float() for index, view in enumerate(teacher_views) if index != anchor_index],
            dim=1,
        ).mean(dim=1)
        fused_teacher = agreement.unsqueeze(1) * fused_teacher + (1.0 - agreement.unsqueeze(1)) * anchor_view
    else:
        agreement = torch.ones((features.shape[0],), device=features.device, dtype=features.dtype)

    student_t = torch.full(
        (features.shape[0],), int(anchor_timestep), device=features.device, dtype=torch.long
    )
    student_noisy = scheduler.add_noise(features, noise, student_t)
    student_energy = row_relative_energies(
        conditional_reconstruction_energies(
            student,
            student_noisy,
            student_t,
            text_embed,
            candidate_labels,
            features,
            scheduler,
            prediction_type,
        )
    )

    teacher_temperature = max(float(teacher_temperature), 1e-6)
    student_temperature = max(float(student_temperature), 1e-6)
    teacher_probs = torch.softmax(-fused_teacher / teacher_temperature, dim=1)
    student_log_probs = torch.log_softmax(-student_energy / student_temperature, dim=1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * student_temperature**2
    teacher_gap = fused_teacher[:, 1:].mean(dim=1) - fused_teacher[:, 0]
    student_gap = student_energy[:, 1:].mean(dim=1) - student_energy[:, 0]
    gap_loss = F.huber_loss(student_gap, teacher_gap, delta=max(float(gap_delta), 1e-6))
    return kl_loss, gap_loss, agreement.mean()


def build_visual_ambiguity_weights(
    config: dict,
    seen_labels: np.ndarray | list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build seen-only visual-confusability negative weights.

    Nearby frozen class means receive a smaller contrastive denominator weight.
    This changes repulsion strength only; it never creates a text-derived soft
    label or imposes a visual ordering on unseen classes.
    """
    root = repo_path(config["train_feeder_args"]["path"])
    raw_features = np.load(root / "train.npy").astype(np.float64)
    features = raw_features.reshape(len(raw_features), -1)
    labels = np.load(root / "train_label.npy").astype(np.int64)
    classes = np.asarray(sorted(int(value) for value in seen_labels), dtype=np.int64)
    means = np.stack([features[labels == label].mean(axis=0) for label in classes])
    pooled_variance = features.var(axis=0, ddof=1).clip(min=1e-8)
    differences = means[:, None, :] - means[None, :, :]
    distances = (np.square(differences) / pooled_variance[None, None, :]).mean(axis=-1)
    nonzero = distances[~np.eye(len(classes), dtype=bool)]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    scale = max(scale, 1e-8)
    minimum = min(max(float(config.get("da_cnce_min_negative_weight", 0.10)), 0.0), 1.0)
    weights = 1.0 - np.exp(-distances / scale)
    weights = np.clip(weights, minimum, 1.0)
    np.fill_diagonal(weights, 1.0)
    class_count = int(max(classes.max() + 1, 1))
    matrix = np.ones((class_count, class_count), dtype=np.float32)
    matrix[np.ix_(classes, classes)] = weights.astype(np.float32)
    return torch.from_numpy(matrix).to(device=device)


def projected_reconstruction_infonce_loss(
    model: DiT,
    aux: C2UAuxHeads,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    negative_labels: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
    tau: float,
) -> torch.Tensor:
    candidates = torch.cat([labels[:, None], negative_labels], dim=1)
    logits: list[torch.Tensor] = []
    for col in range(candidates.shape[1]):
        cand_labels = candidates[:, col]
        fc, fl = split_text_condition(text_embed, cand_labels)
        pred = model(noisy, timesteps, fc, fl)
        pred_sample = prediction_to_sample(pred, noisy, timesteps.long(), scheduler, prediction_type)
        q_s = aux.project_feature(pred_sample.flatten(1))
        q_t = aux.project_text(fc)
        logits.append(torch.sum(q_s * q_t, dim=1))
    logit_matrix = torch.stack(logits, dim=1) / float(tau)
    targets = torch.zeros((labels.shape[0],), device=labels.device, dtype=torch.long)
    return F.cross_entropy(logit_matrix, targets)


def estimate_shared_precision(
    config: dict,
    unseen_labels: np.ndarray,
    shrinkage: float,
    device: torch.device,
) -> torch.Tensor:
    """Pooled within-class precision (Sigma^-1) from SEEN training features.

    Used by the DCR loss so the hard-pair ranking is computed in a discriminative
    (whitened) subspace instead of the shared-variance-dominated raw MSE space.
    Scaled by mean variance so distances stay on the raw-MSE scale.
    """
    train_path = repo_path(config["train_feeder_args"]["path"])
    x = np.load(train_path / "train.npy").astype(np.float64)
    y = np.load(train_path / "train_label.npy").astype(np.int64)
    unseen = set(int(v) for v in unseen_labels.tolist())
    centered = []
    for c in np.unique(y):
        if int(c) in unseen:
            continue
        xc = x[y == c]
        if len(xc) > 1:
            centered.append(xc - xc.mean(axis=0, keepdims=True))
    w = np.concatenate(centered, axis=0)
    d = w.shape[1]
    sigma = np.cov(w, rowvar=False)
    mean_var = float(np.trace(sigma) / d)
    p_full = np.linalg.inv(sigma + float(shrinkage) * mean_var * np.eye(d)) * mean_var
    return torch.from_numpy(p_full).to(device=device, dtype=torch.float32)


def whitened_distance(pred: torch.Tensor, target: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    r = (pred - target).flatten(1)
    return ((r @ precision) * r).sum(dim=1) / r.shape[1]


def discriminative_counterfactual_loss(
    pred_pos: torch.Tensor,
    pred_hard: torch.Tensor,
    target: torch.Tensor,
    precision: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """DCR: hard-pair reconstruction ranking in the whitened discriminative space.

    Forces the correct-text reconstruction to match the true feature better than
    the near-duplicate (hard) text reconstruction along the directions where the
    two classes actually differ (shared 'sitting/desk' variance is down-weighted
    by Sigma^-1). This directly attacks 'reconstructions do not diverge'.
    """
    pos_w = whitened_distance(pred_pos, target, precision)
    hard_w = whitened_distance(pred_hard, target, precision)
    return torch.clamp(pos_w - hard_w + float(margin), min=0.0).mean()


def load_force_hard_pairs(config: dict) -> dict[int, list[int]]:
    pairs = config.get("force_hard_pairs", {})
    output: dict[int, list[int]] = {}
    if isinstance(pairs, dict):
        for key, values in pairs.items():
            if isinstance(values, (list, tuple)):
                output[int(key)] = [int(item) for item in values]
            else:
                output[int(key)] = [int(values)]
    return output


def text_alignment_loss(
    aux: C2UAuxHeads,
    hidden: torch.Tensor,
    text: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    q_s = aux.project_hidden(hidden)
    q_t = aux.project_text(text)
    logits = q_s @ q_t.T / tau
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_mask = labels[:, None].eq(labels[None, :]).float()
    return -((log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)).mean()


def frozen_gallery_match_loss(
    aux: C2UAuxHeads,
    features: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced full seen-gallery matching on frozen skeleton/text inputs.

    The gallery is all observed training classes, so every class competes in
    every update.  Text encoders and skeleton features remain fixed; gradients
    only update the F1 matcher heads.
    """
    gallery_values = sorted({int(value) for value in seen_labels})
    if len(gallery_values) < 2:
        raise ValueError("F1 frozen-gallery matching requires at least two seen classes")
    gallery = torch.as_tensor(gallery_values, device=labels.device, dtype=torch.long)
    gallery_text = pooled_text_embed(text_embed)[gallery]
    logits, csv_gate = aux.frozen_gallery_match_logits(features.flatten(1), gallery_text)
    label_to_column = {label: column for column, label in enumerate(gallery_values)}
    try:
        targets = torch.as_tensor(
            [label_to_column[int(value)] for value in labels.detach().cpu().tolist()],
            device=labels.device,
            dtype=torch.long,
        )
    except KeyError as error:
        raise ValueError(f"F1 target label {int(error.args[0])} is absent from the seen gallery") from error
    tau = max(float(temperature), 1e-6)
    return F.cross_entropy(logits / tau, targets) * tau, csv_gate.mean()


def multipositive_bidirectional_alignment_loss(
    aux: C2UAuxHeads,
    hidden: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    tau: float,
    snr_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced CSV/LLM multi-positive alignment with timestep confidence."""
    tau = max(float(tau), 1e-6)
    pooled_text = pooled_text_embed(text_embed).float()
    if pooled_text.shape[-1] % 2 != 0:
        raise ValueError(
            "C8 CSV/LLM multi-positive alignment requires an even concatenated text dimension"
        )
    half = pooled_text.shape[-1] // 2
    zeros = torch.zeros_like(pooled_text[:, :half])
    csv_view = torch.cat([pooled_text[:, :half], zeros], dim=-1)
    llm_view = torch.cat([zeros, pooled_text[:, half:]], dim=-1)

    q_s = aux.project_hidden(hidden.float())
    seen = torch.as_tensor(seen_labels, device=labels.device, dtype=torch.long)
    gallery_text = torch.cat([csv_view[seen], llm_view[seen]], dim=0)
    gallery_labels = torch.cat([seen, seen], dim=0)
    q_t_gallery = aux.project_text(gallery_text)

    signal_weight = scheduler.alphas_cumprod.to(
        device=labels.device,
        dtype=q_s.dtype,
    )[timesteps.long()].clamp_min(float(snr_floor))
    unique_labels, inverse, class_counts = torch.unique(
        labels,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    class_balance = class_counts[inverse].to(dtype=q_s.dtype).reciprocal()
    anchor_weight = signal_weight * class_balance

    s2t_logits = q_s @ q_t_gallery.T / tau
    s2t_log_prob = F.log_softmax(s2t_logits, dim=1)
    s2t_positive = labels[:, None].eq(gallery_labels[None, :]).to(dtype=q_s.dtype)
    s2t_per_sample = -(
        s2t_log_prob * s2t_positive
    ).sum(dim=1) / s2t_positive.sum(dim=1).clamp_min(1.0)
    s2t_loss = (s2t_per_sample * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-6)

    batch_text = torch.cat([csv_view[unique_labels], llm_view[unique_labels]], dim=0)
    batch_text_labels = torch.cat([unique_labels, unique_labels], dim=0)
    q_t_batch = aux.project_text(batch_text)
    t2s_logits = q_t_batch @ q_s.T / tau
    t2s_logits = t2s_logits + signal_weight.clamp_min(1e-6).log().unsqueeze(0)
    t2s_log_prob = F.log_softmax(t2s_logits, dim=1)
    t2s_positive = batch_text_labels[:, None].eq(labels[None, :]).to(dtype=q_s.dtype)
    t2s_target = t2s_positive * signal_weight.unsqueeze(0)
    t2s_target = t2s_target / t2s_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    t2s_loss = -(t2s_target * t2s_log_prob).sum(dim=1).mean()

    # Keep the auxiliary term in reconstruction-distance units.
    loss = 0.5 * (s2t_loss + t2s_loss) * tau
    return loss, signal_weight.mean()


def neutral_debiased_alignment_loss(
    aux: C2UAuxHeads,
    hidden: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    tau: float,
    negative_floor: float,
    negative_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a class-neutral skeleton query while softly debiasing semantic negatives."""
    tau = max(float(tau), 1e-6)
    negative_floor = min(max(float(negative_floor), 1e-6), 1.0)
    negative_power = max(float(negative_power), 0.0)
    pooled_text = pooled_text_embed(text_embed).float()
    seen = torch.as_tensor(seen_labels, device=labels.device, dtype=torch.long)

    q_s = aux.project_hidden(hidden.float())
    q_t = aux.project_text(pooled_text[seen])
    logits = q_s @ q_t.T / tau

    semantic_text = F.normalize(pooled_text, dim=-1)
    semantic_similarity = semantic_text[labels] @ semantic_text[seen].T
    negative_weight = (1.0 - semantic_similarity).clamp(
        min=negative_floor,
        max=1.0,
    ).pow(negative_power)
    positive_mask = labels[:, None].eq(seen[None, :])
    if not bool(positive_mask.any(dim=1).all()):
        raise RuntimeError("C9 neutral alignment gallery is missing a training label")
    denominator_weight = torch.where(
        positive_mask,
        torch.ones_like(negative_weight),
        negative_weight,
    )
    debiased_logits = logits + denominator_weight.clamp_min(1e-6).log()
    targets = positive_mask.to(dtype=torch.long).argmax(dim=1)

    # Multiplication by tau keeps projector gradients comparable across temperatures.
    loss = F.cross_entropy(debiased_logits, targets) * tau
    negative_only = negative_weight.masked_select(~positive_mask)
    mean_negative_weight = (
        negative_only.mean() if negative_only.numel() else torch.ones((), device=labels.device)
    )
    return loss, mean_negative_weight


def build_action_primitive_context(
    config: dict,
    text_embed: torch.Tensor,
    seen_labels: np.ndarray | list[int],
    device: torch.device,
) -> dict:
    """Build shared motion-word prototypes from the existing frozen CSV text features."""
    if text_embed.dim() != 3 or text_embed.shape[1] < 2:
        raise ValueError("Action primitives require token-level CLIP text features")
    if text_embed.shape[-1] % 2 != 0:
        raise ValueError("Action primitives require concatenated CSV/LLM text features")

    class_list_path = repo_path(config["text_class_list_path"])
    prompts = pd.read_csv(class_list_path)["label"].astype(str).tolist()
    if len(prompts) != int(text_embed.shape[0]):
        raise ValueError(
            f"Primitive prompt count {len(prompts)} does not match text classes {text_embed.shape[0]}"
        )

    tokenizer_path = config.get(
        "text_tokenizer_name_or_path",
        config.get("pretrained_model_name_or_path", "sd2-community/stable-diffusion-2-1"),
    )
    hf_token = HfFolder.get_token()
    hf_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_path,
        subfolder="tokenizer",
        local_files_only=bool(config.get("local_files_only", True)),
        **hf_kwargs,
    )
    tokenized = tokenizer(
        [prompt.strip() for prompt in prompts],
        padding="max_length",
        max_length=int(text_embed.shape[1] - 1),
        truncation=True,
        return_tensors="pt",
    )
    token_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask.bool()
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    default_stopwords = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "from",
        "then", "with", "by", "into", "is", "are", "be", "make", "makes", "around",
        "other", "person", "one", "two", "so", "over", "get", "keep", "'s",
    }
    stopwords = default_stopwords | {
        str(value).strip().lower() for value in config.get("primitive_stopwords", [])
    }

    token_names: dict[int, str] = {}
    class_token_ids: list[set[int]] = []
    for class_idx in range(len(prompts)):
        values: set[int] = set()
        for position, token_id_value in enumerate(token_ids[class_idx].tolist()):
            token_id = int(token_id_value)
            if not bool(attention_mask[class_idx, position]) or token_id in special_ids:
                continue
            token_name = tokenizer.convert_ids_to_tokens(token_id).lower().replace("</w>", "").strip()
            if (
                len(token_name) < 2
                or token_name in stopwords
                or not any(character.isalpha() for character in token_name)
            ):
                continue
            values.add(token_id)
            token_names[token_id] = token_name
        class_token_ids.append(values)

    seen = sorted(int(value) for value in seen_labels)
    seen_df: dict[int, int] = {}
    for token_id in set().union(*(class_token_ids[class_idx] for class_idx in seen)):
        seen_df[token_id] = sum(token_id in class_token_ids[class_idx] for class_idx in seen)
    min_seen_df = max(1, int(config.get("primitive_min_seen_df", 1)))
    max_seen_fraction = min(max(float(config.get("primitive_max_seen_fraction", 0.8)), 0.0), 1.0)
    primitive_ids = [
        token_id
        for token_id, count in seen_df.items()
        if count >= min_seen_df and count / max(1, len(seen)) <= max_seen_fraction
    ]
    primitive_ids.sort(key=lambda token_id: (-seen_df[token_id], token_names[token_id], token_id))
    primitive_ids = primitive_ids[: max(1, int(config.get("primitive_max_count", 256)))]
    if not primitive_ids:
        raise RuntimeError("No action primitives survived the configured seen-class filters")

    primitive_to_col = {token_id: col for col, token_id in enumerate(primitive_ids)}
    class_targets = torch.zeros((len(prompts), len(primitive_ids)), dtype=torch.float32)
    for class_idx, values in enumerate(class_token_ids):
        for token_id in values:
            col = primitive_to_col.get(token_id)
            if col is not None:
                class_targets[class_idx, col] = 1.0

    csv_dim = int(text_embed.shape[-1] // 2)
    csv_tokens = text_embed[:, :-1, :csv_dim].detach().float().cpu()
    prototypes = []
    for token_id in primitive_ids:
        occurrence_mask = token_ids.eq(token_id) & attention_mask
        occurrences = csv_tokens[occurrence_mask]
        if occurrences.numel() == 0:
            raise RuntimeError(f"Primitive token {token_id} has no cached CSV embedding")
        prototypes.append(F.normalize(occurrences.mean(dim=0), dim=0))
    primitive_prototypes = torch.stack(prototypes).to(device=device)

    seen_df_tensor = torch.tensor(
        [seen_df[token_id] for token_id in primitive_ids],
        dtype=torch.float32,
        device=device,
    )
    idf = torch.log((float(len(seen)) + 1.0) / (seen_df_tensor + 1.0)) + 1.0
    class_targets = class_targets.to(device=device)
    pu_class_prior = class_targets[seen].mean(dim=0).clamp(0.0, 1.0)
    raw_token_count = torch.tensor(
        [max(1, len(values)) for values in class_token_ids],
        dtype=torch.float32,
        device=device,
    )
    support_ratio = class_targets.sum(dim=1) / raw_token_count
    class_idf = (class_targets * idf.unsqueeze(0)).sum(dim=1) / class_targets.sum(
        dim=1
    ).clamp_min(1.0)
    idf_span = (idf.max() - idf.min()).clamp_min(1e-6)
    idf_reliability = ((class_idf - idf.min()) / idf_span).clamp(0.0, 1.0)
    # A class is reliable only when its text primitives are represented in the
    # seen vocabulary; rare supported primitives receive a modest preference.
    class_reliability = (support_ratio * (0.5 + 0.5 * idf_reliability)).clamp(0.0, 1.0)
    counterfactual_positions: list[list[int]] = []
    for class_idx, values in enumerate(class_token_ids):
        candidates = [primitive_to_col[token_id] for token_id in values if token_id in primitive_to_col]
        if not candidates:
            counterfactual_positions.append([])
            continue
        best_col = max(candidates, key=lambda col: float(idf[col].item()))
        best_token_id = primitive_ids[best_col]
        counterfactual_positions.append(
            [position for position, token_id in enumerate(token_ids[class_idx].tolist()) if token_id == best_token_id]
        )
    unseen = sorted(set(range(len(prompts))) - set(seen))
    unseen_counts = class_targets[unseen].sum(dim=1) if unseen else torch.empty(0, device=device)
    return {
        "prototypes": primitive_prototypes,
        "class_targets": class_targets,
        "pu_class_prior": pu_class_prior,
        "class_reliability": class_reliability,
        "counterfactual_positions": counterfactual_positions,
        "idf": idf,
        "seen_labels": torch.tensor(seen, device=device, dtype=torch.long),
        "names": [token_names[token_id] for token_id in primitive_ids],
        "description": (
            f"primitives={len(primitive_ids)} source=csv seen_df>={min_seen_df} "
            f"seen_fraction<={max_seen_fraction:.2f} "
            f"unseen_overlap={float(unseen_counts.min().item()) if unseen_counts.numel() else 0.0:.1f}/"
            f"{float(unseen_counts.mean().item()) if unseen_counts.numel() else 0.0:.1f}/"
            f"{float(unseen_counts.max().item()) if unseen_counts.numel() else 0.0:.1f}"
        ),
    }


def counterfactual_text_condition(
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one high-IDF CSV token while preserving global and LLM conditions."""
    if text_embed.dim() != 3 or text_embed.shape[-1] % 2 != 0:
        raise ValueError("Counterfactual token masking requires concatenated token-level text features")
    condition = text_embed[labels].clone()
    local = condition[:, :-1, :]
    csv_dim = local.shape[-1] // 2
    for row, label in enumerate(labels.detach().cpu().tolist()):
        positions = primitive_context["counterfactual_positions"][int(label)]
        for position in positions:
            if position < local.shape[1]:
                local[row, position, :csv_dim] = 0.0
    return condition[:, -1, :], local


def action_primitive_logits(
    aux: C2UAuxHeads,
    hidden_tokens: torch.Tensor,
    primitive_context: dict,
    logit_scale: float,
    logit_bias: float,
) -> torch.Tensor:
    return action_primitive_token_logits(
        aux,
        hidden_tokens,
        primitive_context,
        logit_scale,
        logit_bias,
    ).amax(dim=1)


def action_primitive_token_logits(
    aux: C2UAuxHeads,
    hidden_tokens: torch.Tensor,
    primitive_context: dict,
    logit_scale: float,
    logit_bias: float,
) -> torch.Tensor:
    """Return a DiT-token by text-primitive compatibility matrix."""
    if hidden_tokens.dim() != 3:
        raise ValueError(f"Expected latent motion tokens shaped (B, K, D), got {tuple(hidden_tokens.shape)}")
    projected_tokens = aux.project_hidden(hidden_tokens.float())
    prototypes = primitive_context["prototypes"].to(
        device=projected_tokens.device,
        dtype=projected_tokens.dtype,
    )
    token_similarity = torch.einsum("bkd,pd->bkp", projected_tokens, prototypes)
    return token_similarity * float(logit_scale) + float(logit_bias)


def action_primitive_multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
    negative_gamma: float,
    negative_weight: float,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = primitive_context["class_targets"][labels].to(dtype=logits.dtype)
    idf = primitive_context["idf"].to(dtype=logits.dtype).unsqueeze(0)
    positive_weights = targets * idf
    positive_loss = (
        F.softplus(-logits) * positive_weights
    ).sum(dim=1) / positive_weights.sum(dim=1).clamp_min(1.0)

    negative_mask = 1.0 - targets
    negative_focal = torch.sigmoid(logits).pow(max(float(negative_gamma), 0.0))
    negative_loss = (
        F.softplus(logits) * negative_focal * negative_mask
    ).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1.0)
    per_sample = positive_loss + float(negative_weight) * negative_loss
    if sample_weight is not None:
        if sample_weight.shape != per_sample.shape:
            raise ValueError("Primitive sample_weight must have shape (B,)")
        loss = (per_sample * sample_weight.to(dtype=per_sample.dtype)).mean()
    else:
        loss = per_sample.mean()
    return loss, targets.mean()


def action_primitive_nnpu_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
    prior_floor: float,
    prior_ceiling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-negative PU risk for incomplete text-derived primitive labels.

    A primitive named by a class prompt is an observed positive. Its absence is
    unlabeled rather than a hard negative because a prompt can omit a valid
    motion component. The seen-class prior corrects the unlabeled risk.
    """
    targets = primitive_context["class_targets"][labels].to(dtype=logits.dtype)
    positives = targets.gt(0.0)
    positive_count = positives.sum(dim=0)
    valid = positive_count.gt(0)
    if not bool(valid.any()):
        raise RuntimeError("nnPU batch contains no observed positive primitive")

    positive_loss = F.softplus(-logits)
    negative_loss = F.softplus(logits)
    positive_count_safe = positive_count.clamp_min(1).to(dtype=logits.dtype)
    risk_positive = (positive_loss * positives).sum(dim=0) / positive_count_safe
    risk_positive_as_negative = (negative_loss * positives).sum(dim=0) / positive_count_safe
    risk_unlabeled_as_negative = negative_loss.mean(dim=0)

    lower = min(max(float(prior_floor), 0.0), 1.0)
    upper = min(max(float(prior_ceiling), lower), 1.0)
    prior = primitive_context["pu_class_prior"].to(dtype=logits.dtype).clamp(lower, upper)
    corrected_negative = risk_unlabeled_as_negative - prior * risk_positive_as_negative
    per_primitive_risk = prior * risk_positive + corrected_negative.clamp_min(0.0)
    weights = primitive_context["idf"].to(dtype=logits.dtype) * valid.to(dtype=logits.dtype)
    loss = (per_primitive_risk * weights).sum() / weights.sum().clamp_min(1.0)
    return (
        loss,
        targets.mean(),
        valid.to(dtype=logits.dtype).mean(),
        corrected_negative[valid].mean(),
    )


def semi_uot_primitive_coverage_loss(
    token_logits: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
    epsilon: float,
    target_tau: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable semi-UOT from DiT motion tokens to true text primitives.

    The visual-token marginal is fixed, preventing loss reduction through
    visual mass deletion.  The positive text-primitive marginal is KL-relaxed,
    since an action name can include an abstract primitive absent from one
    motion token.  It is used only on observed seen labels during training.
    """
    if token_logits.dim() != 3:
        raise ValueError(f"Expected primitive token logits (B, K, P), got {tuple(token_logits.shape)}")
    epsilon = max(float(epsilon), 1e-6)
    target_tau = max(float(target_tau), 0.0)
    iterations = max(1, int(iterations))
    target, valid = primitive_class_measures(primitive_context, labels)
    if not bool(valid.all()):
        raise RuntimeError("Semi-UOT requires every seen class to have positive primitive support")
    target = target.to(device=token_logits.device, dtype=token_logits.dtype)
    batch, num_tokens, _ = token_logits.shape
    source = torch.full(
        (batch, num_tokens), 1.0 / float(num_tokens), device=token_logits.device, dtype=token_logits.dtype
    )
    costs = F.softplus(-token_logits)
    log_kernel = -costs / epsilon
    log_source = source.log()
    target_support = target.gt(0.0)
    log_target = torch.where(target_support, target.log(), torch.full_like(target, float("-inf")))
    log_v = torch.where(target_support, torch.zeros_like(target), torch.full_like(target, float("-inf")))
    target_exponent = target_tau / (target_tau + epsilon) if target_tau > 0.0 else 0.0
    for _ in range(iterations):
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        if target_exponent == 0.0:
            log_v = torch.where(target_support, torch.zeros_like(log_v), torch.full_like(log_v, float("-inf")))
        else:
            log_column_mass = torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
            log_v = target_exponent * (log_target - log_column_mass)
            log_v = torch.where(target_support, log_v, torch.full_like(log_v, float("-inf")))
    log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
    log_plan = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
    plan = torch.exp(log_plan)
    plan_safe = torch.where(target_support.unsqueeze(1), plan, torch.zeros_like(plan))
    reference = source.unsqueeze(2) * target.unsqueeze(1)
    log_reference = torch.where(
        target_support.unsqueeze(1),
        log_source.unsqueeze(2) + log_target.unsqueeze(1),
        torch.zeros_like(log_plan),
    )
    log_plan_safe = plan_safe.clamp_min(1e-38).log()
    kl_plan = (plan_safe * (log_plan_safe - log_reference) - plan_safe + reference).sum(dim=(1, 2))
    column_mass = plan_safe.sum(dim=1)
    kl_target = torch.where(
        target_support,
        column_mass * (column_mass.clamp_min(1e-38).log() - log_target) - column_mass + target,
        torch.zeros_like(column_mass),
    ).sum(dim=1)
    objective = (plan_safe * costs).sum(dim=(1, 2)) + epsilon * kl_plan + target_tau * kl_target
    target_l1 = (column_mass - target).abs().sum(dim=1).mean()
    return objective.mean(), target_l1


def primitive_feature_distribution_loss(
    aux: C2UAuxHeads,
    features: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """C17 conditional Gaussian feature likelihood without pairwise negatives."""
    primitive_targets = primitive_context["class_targets"][labels].to(
        device=features.device,
        dtype=features.dtype,
    )
    mean, log_variance = aux.primitive_feature_distribution_parameters(primitive_targets)
    target = features.flatten(1).detach()
    inverse_variance = torch.exp(-log_variance)
    nll = 0.5 * ((target - mean).square() * inverse_variance + log_variance)
    return nll.mean(), F.mse_loss(mean, target), torch.exp(log_variance).mean()


def build_seen_visual_prototype_context(
    config: dict,
    seen_labels: np.ndarray | list[int],
    text_class_count: int,
    device: torch.device,
) -> dict:
    """Class prototypes and a shared within-class variance from frozen features."""
    train_path = repo_path(config["train_feeder_args"]["path"])
    features = np.load(train_path / "train.npy").astype(np.float32)
    labels = np.load(train_path / "train_label.npy").astype(np.int64)
    seen = np.asarray(sorted(int(label) for label in seen_labels), dtype=np.int64)
    prototypes = []
    residuals = []
    for label in seen:
        class_features = features[labels == label]
        if class_features.size == 0:
            raise RuntimeError(f"Missing training features for seen class {int(label)}")
        prototype = class_features.mean(axis=0)
        prototypes.append(prototype)
        residuals.append(class_features - prototype)
    prototype_tensor = torch.from_numpy(np.stack(prototypes)).to(device=device)
    shared_variance = torch.from_numpy(
        np.concatenate(residuals, axis=0).var(axis=0).astype(np.float32)
    ).to(device=device).clamp_min(1e-4)
    class_index = torch.full((text_class_count,), -1, device=device, dtype=torch.long)
    class_index[torch.as_tensor(seen, device=device, dtype=torch.long)] = torch.arange(
        len(seen), device=device, dtype=torch.long
    )
    return {
        "seen_labels": torch.as_tensor(seen, device=device, dtype=torch.long),
        "prototypes": prototype_tensor,
        "shared_variance": shared_variance,
        "class_index": class_index,
    }


def episodic_prototype_completion_loss(
    aux: C2UAuxHeads,
    labels: torch.Tensor,
    primitive_context: dict,
    visual_context: dict,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LOO pseudo-unseen prototype reconstruction using only other seen classes."""
    if aux.episodic_proto_generator is None:
        raise RuntimeError("C19 episodic prototype generator was not initialized")
    class_index = visual_context["class_index"][labels]
    if bool(class_index.lt(0).any()):
        raise RuntimeError("C19 batch contains a label outside the seen prototype context")
    support_primitives = primitive_context["class_targets"][visual_context["seen_labels"]]
    target_primitives = primitive_context["class_targets"][labels]
    predicted = aux.episodic_proto_generator(
        target_primitives,
        support_primitives,
        visual_context["prototypes"],
        heldout_indices=class_index,
        temperature=temperature,
    )
    target = visual_context["prototypes"][class_index].detach()
    mse = F.mse_loss(predicted, target)
    nll = 0.5 * (
        (predicted - target).square() / visual_context["shared_variance"]
        + visual_context["shared_variance"].log()
    ).mean()
    return mse, nll


def episodic_proto_seen_reliability(
    aux: C2UAuxHeads,
    primitive_context: dict,
    visual_context: dict,
    temperature: float,
) -> tuple[float, float]:
    """Seen-only LOO retrieval controls the maximum C19 late-fusion weight."""
    if aux.episodic_proto_generator is None:
        return 0.0, 0.0
    seen_labels = visual_context["seen_labels"]
    support_primitives = primitive_context["class_targets"][seen_labels]
    indices = torch.arange(len(seen_labels), device=seen_labels.device, dtype=torch.long)
    predicted = aux.episodic_proto_generator(
        support_primitives,
        support_primitives,
        visual_context["prototypes"],
        heldout_indices=indices,
        temperature=temperature,
    )
    scores = ((visual_context["prototypes"].unsqueeze(1) - predicted.unsqueeze(0)).square()
              / visual_context["shared_variance"].view(1, 1, -1)).mean(dim=-1)
    accuracy = float((torch.argmin(scores, dim=1) == indices).float().mean().item())
    chance = 1.0 / float(len(seen_labels))
    reliability = max(0.0, min(1.0, (accuracy - chance) / max(1e-6, 1.0 - chance)))
    return accuracy, reliability


def primitive_class_measures(
    primitive_context: dict,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IDF-weighted class measures and an explicit support-validity mask."""
    targets = primitive_context["class_targets"][labels].float()
    idf = primitive_context["idf"].to(device=targets.device, dtype=targets.dtype)
    weighted = targets * idf.unsqueeze(0)
    mass = weighted.sum(dim=1, keepdim=True)
    valid = mass.squeeze(1).gt(0.0)
    measure = torch.where(valid.unsqueeze(1), weighted / mass.clamp_min(1e-8), torch.zeros_like(weighted))
    return measure, valid


@torch.no_grad()
def unbalanced_sinkhorn_cost(
    costs: torch.Tensor,
    source_measure: torch.Tensor,
    target_measure: torch.Tensor,
    epsilon: float,
    tau_source: float,
    tau_target: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-domain KL-relaxed UOT with exact zero target support.

    The input measures need not have identical transported mass.  Zero-valued
    text primitives are represented by ``-inf`` dual variables rather than a
    numerical floor, so absent primitives cannot receive transport mass.
    """
    if costs.dim() != 3:
        raise ValueError(f"Expected UOT costs shaped (N, K, P), got {tuple(costs.shape)}")
    eps = max(float(epsilon), 1e-6)
    tau_s = max(float(tau_source), 0.0)
    tau_t = max(float(tau_target), 0.0)
    if tau_s <= 0.0 or tau_t <= 0.0:
        raise ValueError("UOT marginal penalties must be positive")
    if bool(source_measure.le(0.0).any()) or bool(source_measure.sum(dim=1).le(0.0).any()):
        raise ValueError("UOT source measures must be strictly positive")
    if bool(target_measure.sum(dim=1).le(0.0).any()):
        raise ValueError("UOT target measures must have non-empty support")

    log_kernel = -costs / eps
    log_source = source_measure.log()
    log_target = torch.where(
        target_measure.gt(0.0),
        target_measure.log(),
        torch.full_like(target_measure, float("-inf")),
    )
    rho_s = tau_s / (tau_s + eps)
    rho_t = tau_t / (tau_t + eps)
    log_v = torch.zeros_like(target_measure)
    log_v = torch.where(target_measure.gt(0.0), log_v, torch.full_like(log_v, float("-inf")))
    for _ in range(max(1, int(iterations))):
        log_u = rho_s * (
            log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        )
        log_v = rho_t * (
            log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
        )

    log_plan = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
    plan = torch.exp(log_plan)
    row_mass = plan.sum(dim=2)
    col_mass = plan.sum(dim=1)

    def generalized_kl(value: torch.Tensor, log_value: torch.Tensor, reference: torch.Tensor, log_reference: torch.Tensor) -> torch.Tensor:
        positive_value = value.gt(0.0)
        term = torch.where(
            positive_value,
            value * (log_value - log_reference) - value + reference,
            reference,
        )
        return term.flatten(1).sum(dim=1)

    reference_plan = source_measure.unsqueeze(2) * target_measure.unsqueeze(1)
    log_reference_plan = log_source.unsqueeze(2) + log_target.unsqueeze(1)
    kl_plan = generalized_kl(plan, log_plan, reference_plan, log_reference_plan)
    kl_source = generalized_kl(
        row_mass,
        row_mass.clamp_min(1e-38).log(),
        source_measure,
        log_source,
    )
    kl_target = generalized_kl(
        col_mass,
        col_mass.clamp_min(1e-38).log(),
        target_measure,
        log_target,
    )
    transport = (plan * costs).flatten(1).sum(dim=1)
    objective = transport + eps * kl_plan + tau_s * kl_source + tau_t * kl_target
    return objective, plan.sum(dim=(1, 2)), transport


@torch.no_grad()
def evaluate_uot_probe(
    model: DiT,
    aux: C2UAuxHeads,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    text_embed: torch.Tensor,
    seen_labels: np.ndarray,
    primitive_context: dict,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Check whether frozen C10 tokens support a valid class-conditional UOT energy."""
    model.eval()
    aux.eval()
    candidate_count = max(1, int(config.get("uot_probe_num_negatives", 1)))
    timestep = int(config.get("uot_probe_timestep", config.get("idx_inference_step", 25)))
    epsilon = float(config.get("uot_epsilon", 0.05))
    tau_source = float(config.get("uot_tau_visual", 0.20))
    tau_target = float(config.get("uot_tau_text", 0.20))
    iterations = int(config.get("uot_sinkhorn_iters", 30))
    align_layer = int(config.get("align_layer", 8))
    prototypes = primitive_context["prototypes"].to(device=device, dtype=torch.float32)
    scores: list[torch.Tensor] = []
    raw_objectives: list[torch.Tensor] = []
    transport_terms: list[torch.Tensor] = []
    transported_mass: list[torch.Tensor] = []
    valid_rows = 0
    total_rows = 0
    for features, labels in tqdm(loader, desc="uot probe"):
        features = features.to(device=device, dtype=dtype).unsqueeze(1)
        labels = labels.to(device=device, dtype=torch.long)
        batch = features.shape[0]
        timesteps = torch.full((batch,), timestep, device=device, dtype=torch.long)
        # Fixed zero noise makes the energy diagnostic reproducible and keeps
        # every candidate comparison on exactly the same frozen visual evidence.
        noisy = scheduler.add_noise(features, torch.zeros_like(features), timesteps)
        neutral_fc = torch.zeros((batch, text_embed.shape[-1]), device=device, dtype=dtype)
        _, hidden_states = model(
            noisy,
            timesteps.to(dtype),
            neutral_fc,
            None,
            return_hidden=True,
            hidden_layers=[align_layer],
        )
        if not hidden_states:
            raise RuntimeError("UOT probe could not obtain neutral hidden tokens")
        visual_tokens = aux.project_hidden(hidden_states[0].float())
        negative_labels = sample_infonce_negative_labels(
            labels,
            [int(value) for value in seen_labels],
            num_neg=candidate_count,
            device=device,
        )
        candidates = torch.cat((labels.unsqueeze(1), negative_labels), dim=1)
        measures, candidate_valid = primitive_class_measures(
            primitive_context,
            candidates.reshape(-1),
        )
        expanded_tokens = visual_tokens.unsqueeze(1).expand(-1, candidates.shape[1], -1, -1)
        expanded_tokens = expanded_tokens.reshape(-1, visual_tokens.shape[1], visual_tokens.shape[2])
        costs = 1.0 - torch.einsum("nkd,pd->nkp", expanded_tokens, prototypes)
        source = torch.full(
            (costs.shape[0], costs.shape[1]),
            1.0 / float(costs.shape[1]),
            device=device,
            dtype=costs.dtype,
        )
        valid_indices = candidate_valid.nonzero(as_tuple=False).flatten()
        objective = torch.full((costs.shape[0],), float("nan"), device=device, dtype=costs.dtype)
        mass = torch.full_like(objective, float("nan"))
        transport = torch.full_like(objective, float("nan"))
        if valid_indices.numel():
            valid_objective, valid_mass, valid_transport = unbalanced_sinkhorn_cost(
                costs[valid_indices], source[valid_indices], measures[valid_indices].to(dtype=costs.dtype),
                epsilon=epsilon, tau_source=tau_source, tau_target=tau_target, iterations=iterations,
            )
            objective[valid_indices] = valid_objective
            mass[valid_indices] = valid_mass
            transport[valid_indices] = valid_transport
        objective = objective.view(batch, -1)
        mass = mass.view(batch, -1)
        transport = transport.view(batch, -1)
        score_mode = str(config.get("uot_probe_score_mode", "objective")).lower()
        if score_mode == "objective":
            energy = objective
        elif score_mode in {"mass_normalized_transport", "geometry"}:
            energy = transport / mass.clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported uot_probe_score_mode: {score_mode}")
        row_valid = torch.isfinite(energy).all(dim=1)
        if row_valid.any():
            scores.append(energy[row_valid].detach().float().cpu())
            raw_objectives.append(objective[row_valid].detach().float().cpu())
            transport_terms.append(transport[row_valid].detach().float().cpu())
            transported_mass.append(mass[row_valid].detach().float().cpu())
            valid_rows += int(row_valid.sum().item())
        total_rows += batch
    if not scores:
        raise RuntimeError("UOT probe found no samples with primitive support for all candidates")
    all_scores = torch.cat(scores)
    all_objectives = torch.cat(raw_objectives)
    all_transport = torch.cat(transport_terms)
    all_mass = torch.cat(transported_mass)
    positive = all_scores[:, 0]
    negatives = all_scores[:, 1:]
    gap = negatives.mean(dim=1) - positive
    flat_energy = all_scores.flatten()
    flat_mass = all_mass.flatten()
    centered_energy = flat_energy - flat_energy.mean()
    centered_mass = flat_mass - flat_mass.mean()
    energy_mass_correlation = centered_energy.dot(centered_mass) / (
        centered_energy.norm() * centered_mass.norm()
    ).clamp_min(1e-8)
    return {
        "valid_samples": valid_rows,
        "total_samples": total_rows,
        "valid_fraction": valid_rows / max(1, total_rows),
        "positive_energy": float(positive.mean().item()),
        "negative_energy": float(negatives.mean().item()),
        "positive_lower_fraction": float((positive.unsqueeze(1) < negatives).all(dim=1).float().mean().item()),
        "mean_energy_gap": float(gap.mean().item()),
        "positive_transport_mass": float(all_mass[:, 0].mean().item()),
        "negative_transport_mass": float(all_mass[:, 1:].mean().item()),
        "positive_minus_negative_mass": float(
            (all_mass[:, 0] - all_mass[:, 1:].mean(dim=1)).mean().item()
        ),
        "energy_mass_correlation": float(energy_mass_correlation.item()),
        "raw_positive_objective": float(all_objectives[:, 0].mean().item()),
        "raw_negative_objective": float(all_objectives[:, 1:].mean().item()),
        "positive_transport_cost": float(all_transport[:, 0].mean().item()),
        "negative_transport_cost": float(all_transport[:, 1:].mean().item()),
        "score_mode": str(config.get("uot_probe_score_mode", "objective")).lower(),
        "candidate_count": int(all_scores.shape[1]),
        "uot_epsilon": epsilon,
        "uot_tau_visual": tau_source,
        "uot_tau_text": tau_target,
        "uot_sinkhorn_iters": iterations,
    }


def balanced_sinkhorn_transport(
    scores: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return a uniform-prior, entropically regularized transport plan.

    Rows are test samples and columns are candidate unseen labels.  The solver
    sees only the final score matrix, never test labels.  Log-domain updates
    keep the assignment stable for sharp diffusion-score differences.
    """
    if scores.dim() != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("Sinkhorn OT requires a non-empty (samples, classes) score matrix")
    epsilon = max(float(epsilon), 1e-6)
    iterations = max(1, int(iterations))
    num_samples, num_classes = scores.shape
    log_kernel = -scores.float() / epsilon
    log_row_mass = -float(np.log(float(num_samples)))
    log_class_mass = -float(np.log(float(num_classes)))
    log_u = torch.zeros((num_samples,), device=scores.device, dtype=log_kernel.dtype)
    log_v = torch.zeros((num_classes,), device=scores.device, dtype=log_kernel.dtype)
    for _ in range(iterations):
        log_u = log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_class_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def semi_unbalanced_sinkhorn_transport(
    scores: torch.Tensor,
    epsilon: float,
    target_tau: float,
    iterations: int,
) -> torch.Tensor:
    """Return a row-constrained, target-unbalanced entropic transport plan.

    Every evaluation sample keeps its prescribed row mass while the unseen
    class marginal is only softly attracted to the uniform prior.  This is
    the appropriate relaxation for ZSL: each test feature must receive a
    label, whereas the unknown test-class frequencies need not be uniform.
    """
    if scores.dim() != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("Semi-unbalanced OT requires a non-empty (samples, classes) score matrix")
    epsilon = max(float(epsilon), 1e-6)
    target_tau = max(float(target_tau), 0.0)
    iterations = max(1, int(iterations))
    num_samples, num_classes = scores.shape
    log_kernel = -scores.float() / epsilon
    log_row_mass = -float(np.log(float(num_samples)))
    log_class_mass = -float(np.log(float(num_classes)))
    target_exponent = target_tau / (target_tau + epsilon)
    log_u = torch.zeros((num_samples,), device=scores.device, dtype=log_kernel.dtype)
    log_v = torch.zeros((num_classes,), device=scores.device, dtype=log_kernel.dtype)
    for _ in range(iterations):
        log_u = log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        if target_exponent == 0.0:
            log_v.zero_()
        else:
            log_column_mass = torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
            log_v = target_exponent * (log_class_mass - log_column_mass)

    # Reimpose the hard source marginal after the final target update.
    log_u = log_row_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def action_primitive_debiased_prototype_contrast_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    primitive_context: dict,
    tau: float,
    negative_floor: float,
    negative_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrast primitive sets while downweighting overlapping action negatives."""
    tau = max(float(tau), 1e-6)
    negative_floor = min(max(float(negative_floor), 1e-6), 1.0)
    negative_power = max(float(negative_power), 0.0)
    class_targets = primitive_context["class_targets"].to(dtype=logits.dtype)
    seen_labels = primitive_context["seen_labels"].to(device=labels.device)
    candidate_targets = class_targets[seen_labels]
    target_sets = class_targets[labels]
    idf = primitive_context["idf"].to(dtype=logits.dtype)

    predicted_sets = F.normalize(torch.sigmoid(logits) * idf.unsqueeze(0), dim=1)
    candidate_sets = F.normalize(candidate_targets * idf.unsqueeze(0), dim=1)
    set_logits = predicted_sets @ candidate_sets.T / tau
    intersection = target_sets @ candidate_targets.T
    union = (
        target_sets.sum(dim=1, keepdim=True)
        + candidate_targets.sum(dim=1).unsqueeze(0)
        - intersection
    ).clamp_min(1.0)
    overlap = intersection / union
    negative_weight = (1.0 - overlap).clamp(min=negative_floor, max=1.0).pow(
        negative_power
    )
    positive_mask = labels[:, None].eq(seen_labels[None, :])
    if not bool(positive_mask.any(dim=1).all()):
        raise RuntimeError("Primitive prototype gallery is missing a training label")
    denominator_weight = torch.where(
        positive_mask,
        torch.ones_like(negative_weight),
        negative_weight,
    )
    targets = positive_mask.to(dtype=torch.long).argmax(dim=1)
    loss = F.cross_entropy(set_logits + denominator_weight.log(), targets) * tau
    negative_only = negative_weight.masked_select(~positive_mask)
    mean_negative_weight = (
        negative_only.mean() if negative_only.numel() else torch.ones((), device=logits.device)
    )
    return loss, mean_negative_weight


def tmr_filtered_primitive_text_infonce_loss(
    aux: C2UAuxHeads,
    hidden: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    tau: float,
    similarity_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TMR-style bidirectional InfoNCE with text-similar negatives removed."""
    tau = max(float(tau), 1e-6)
    threshold = min(max(float(similarity_threshold), -1.0), 1.0)
    seen = torch.as_tensor(seen_labels, device=labels.device, dtype=torch.long)
    pooled_text = pooled_text_embed(text_embed).float()
    text_gallery = pooled_text[seen]
    q_s = aux.project_hidden(hidden.float())
    q_t = aux.project_text(text_gallery)
    raw_text = F.normalize(pooled_text, dim=-1)

    s2t_targets = labels[:, None].eq(seen[None, :]).to(dtype=torch.long).argmax(dim=1)
    s2t_allowed = raw_text[labels] @ raw_text[seen].T < threshold
    s2t_allowed.scatter_(1, s2t_targets[:, None], True)
    s2t_logits = (q_s @ q_t.T / tau).masked_fill(~s2t_allowed, -torch.inf)
    s2t_loss = F.cross_entropy(s2t_logits, s2t_targets) * tau

    present_labels = torch.unique(labels, sorted=True)
    present_text_indices = present_labels[:, None].eq(seen[None, :]).to(
        dtype=torch.long
    ).argmax(dim=1)
    t2s_logits = q_t[present_text_indices] @ q_s.T / tau
    t2s_positive = present_labels[:, None].eq(labels[None, :])
    t2s_allowed = raw_text[present_labels] @ raw_text[labels].T < threshold
    t2s_allowed = t2s_allowed | t2s_positive
    t2s_log_prob = F.log_softmax(t2s_logits.masked_fill(~t2s_allowed, -torch.inf), dim=1)
    t2s_loss = -(
        torch.where(t2s_positive, t2s_log_prob, torch.zeros_like(t2s_log_prob))
    ).sum(dim=1).div(t2s_positive.sum(dim=1).clamp_min(1)).mean() * tau

    negative_pairs = ~labels[:, None].eq(seen[None, :])
    filtered_fraction = (
        ((~s2t_allowed) & negative_pairs).sum().to(dtype=q_s.dtype)
        / negative_pairs.sum().clamp_min(1)
    )
    return 0.5 * (s2t_loss + t2s_loss), filtered_fraction


def backward_primitive_pcgrad(
    main_loss: torch.Tensor,
    primitive_loss: torch.Tensor,
    shared_params: list[torch.nn.Parameter],
    auxiliary_params: list[torch.nn.Parameter],
    accumulation_steps: int,
    max_shared_ratio: float,
) -> tuple[float, float, float]:
    """Accumulate main gradients plus a conflict-projected primitive gradient."""
    divisor = float(max(1, accumulation_steps))
    main_grads = torch.autograd.grad(
        main_loss / divisor,
        shared_params,
        allow_unused=True,
    )
    primitive_params = shared_params + auxiliary_params
    primitive_grads = torch.autograd.grad(
        primitive_loss / divisor,
        primitive_params,
        allow_unused=True,
    )
    shared_primitive_grads = primitive_grads[: len(shared_params)]

    dot = torch.zeros((), device=main_loss.device, dtype=torch.float32)
    main_norm_sq = torch.zeros_like(dot)
    primitive_norm_sq = torch.zeros_like(dot)
    for main_grad, primitive_grad in zip(main_grads, shared_primitive_grads):
        if main_grad is not None:
            main_norm_sq += main_grad.detach().float().square().sum()
        if primitive_grad is not None:
            primitive_norm_sq += primitive_grad.detach().float().square().sum()
        if main_grad is not None and primitive_grad is not None:
            dot += (main_grad.detach().float() * primitive_grad.detach().float()).sum()

    eps = 1e-12
    main_norm = main_norm_sq.sqrt()
    conflict = bool(dot.item() < 0.0 and main_norm_sq.item() > eps)
    projection_coefficient = dot / main_norm_sq.clamp_min(eps) if conflict else dot.new_zeros(())
    projected_norm_sq = (
        primitive_norm_sq - dot.square() / main_norm_sq.clamp_min(eps)
        if conflict
        else primitive_norm_sq
    ).clamp_min(0.0)
    projected_norm = projected_norm_sq.sqrt()
    allowed_norm = max(float(max_shared_ratio), 0.0) * main_norm
    shared_scale = min(
        1.0,
        float((allowed_norm / projected_norm.clamp_min(eps)).item()),
    )
    denominator = (main_norm * primitive_norm_sq.sqrt()).clamp_min(eps)
    cosine = float((dot / denominator).item()) if primitive_norm_sq.item() > eps else 0.0

    with torch.no_grad():
        projection_value = float(projection_coefficient.item()) if conflict else 0.0
        for parameter, main_grad, primitive_grad in zip(shared_params, main_grads, shared_primitive_grads):
            if main_grad is not None:
                if parameter.grad is None:
                    parameter.grad = main_grad.detach().clone()
                else:
                    parameter.grad.add_(main_grad.detach())
            if primitive_grad is not None and shared_scale > 0.0:
                if parameter.grad is None:
                    parameter.grad = primitive_grad.detach().clone().mul_(shared_scale)
                else:
                    parameter.grad.add_(primitive_grad.detach(), alpha=shared_scale)
                if conflict and main_grad is not None:
                    parameter.grad.add_(
                        main_grad.detach(),
                        alpha=-shared_scale * projection_value,
                    )

        auxiliary_grads = primitive_grads[len(shared_params) :]
        for parameter, gradient in zip(auxiliary_params, auxiliary_grads):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.add_(gradient.detach())

    return cosine, shared_scale, float(conflict)


def action_primitive_class_scores(
    logits: torch.Tensor,
    candidate_labels: np.ndarray | list[int],
    primitive_context: dict,
    negative_weight: float,
    score_mode: str = "c10",
    pu_positive_probability: float = 0.95,
) -> torch.Tensor:
    candidates = torch.as_tensor(candidate_labels, device=logits.device, dtype=torch.long)
    targets = primitive_context["class_targets"][candidates].to(dtype=logits.dtype)
    idf = primitive_context["idf"].to(dtype=logits.dtype)
    positive_template = targets * idf.unsqueeze(0)
    positive_scores = F.softplus(-logits) @ positive_template.T
    positive_scores = positive_scores / positive_template.sum(dim=1).clamp_min(1.0).unsqueeze(0)

    score_mode = score_mode.lower()
    if score_mode in {"positive_only", "pu_positive"}:
        # No vocabulary overlap means no semantic evidence. Use the row mean so
        # row z-scoring leaves such a candidate neutral rather than rewarding
        # its empty primitive template with an artificial zero cost.
        supported = positive_template.sum(dim=1).gt(0.0)
        if bool(supported.any()):
            fallback = positive_scores[:, supported].mean(dim=1, keepdim=True)
            positive_scores = torch.where(supported.unsqueeze(0), positive_scores, fallback)
        return positive_scores
    if score_mode == "pu_soft":
        # Text presence is strong but imperfect positive evidence. Text absence
        # is assigned the seen-class primitive prior instead of a hard zero,
        # retaining calibrated class exclusion without false-negative pressure.
        positive_probability = min(max(float(pu_positive_probability), 0.0), 1.0)
        prior = primitive_context["pu_class_prior"].to(dtype=logits.dtype).unsqueeze(0)
        posterior = targets * positive_probability + (1.0 - targets) * prior
        primitive_cost = (
            posterior.unsqueeze(0) * F.softplus(-logits).unsqueeze(1)
            + (1.0 - posterior).unsqueeze(0) * F.softplus(logits).unsqueeze(1)
        )
        soft_scores = (primitive_cost * idf.view(1, 1, -1)).sum(dim=-1)
        soft_scores = soft_scores / idf.sum().clamp_min(1.0)
        supported = targets.sum(dim=1).gt(0.0)
        if bool(supported.any()):
            fallback = soft_scores[:, supported].mean(dim=1, keepdim=True)
            soft_scores = torch.where(supported.unsqueeze(0), soft_scores, fallback)
        return soft_scores
    if score_mode not in {"c10", "hard_negative"}:
        raise ValueError(f"Unsupported primitive score mode: {score_mode}")

    negative_template = (1.0 - targets) * idf.unsqueeze(0)
    negative_scores = F.softplus(logits) @ negative_template.T
    negative_scores = negative_scores / negative_template.sum(dim=1).clamp_min(1.0).unsqueeze(0)
    return positive_scores + float(negative_weight) * negative_scores


def primitive_risk_gate_features(
    a0_scores: torch.Tensor,
    primitive_scores: torch.Tensor,
) -> torch.Tensor:
    """Features shared by the held-out gate fitter and evaluation-time gate."""
    if a0_scores.shape != primitive_scores.shape or a0_scores.dim() != 2:
        raise ValueError("Primitive risk gate requires matching score matrices shaped (N, C)")
    if a0_scores.shape[1] < 2:
        zeros = torch.zeros((a0_scores.shape[0],), device=a0_scores.device, dtype=a0_scores.dtype)
        return torch.stack((zeros, zeros, zeros), dim=1)
    a0_sorted = torch.sort(a0_scores, dim=1).values
    primitive_sorted = torch.sort(primitive_scores, dim=1).values
    a0_margin = a0_sorted[:, 1] - a0_sorted[:, 0]
    primitive_margin = primitive_sorted[:, 1] - primitive_sorted[:, 0]
    agreement = torch.argmin(a0_scores, dim=1).eq(
        torch.argmin(primitive_scores, dim=1)
    ).to(dtype=a0_scores.dtype)
    return torch.stack((a0_margin, primitive_margin, agreement), dim=1)


def load_primitive_risk_gate(path_value: str | Path, device: torch.device) -> dict:
    path = repo_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"feature_names", "mean", "scale", "coef", "intercept"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Primitive risk gate {path} is missing keys: {sorted(missing)}")
    expected_names = ["a0_margin", "primitive_margin", "top1_agreement"]
    if payload["feature_names"] != expected_names:
        raise ValueError("Primitive risk gate feature schema does not match this evaluator")
    mean = torch.tensor(payload["mean"], device=device, dtype=torch.float32)
    scale = torch.tensor(payload["scale"], device=device, dtype=torch.float32).clamp_min(1e-6)
    coef = torch.tensor(payload["coef"], device=device, dtype=torch.float32)
    if mean.numel() != len(expected_names) or scale.numel() != len(expected_names) or coef.numel() != len(expected_names):
        raise ValueError("Primitive risk gate parameter dimensions are invalid")
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": float(payload["intercept"])}


def primitive_utility_gate_features(
    a0_scores: torch.Tensor,
    primitive_scores: torch.Tensor,
    a0_candidates: torch.Tensor,
    nominal_candidates: torch.Tensor,
) -> torch.Tensor:
    """Features for predicting whether the fixed primitive action has utility."""
    base = primitive_risk_gate_features(a0_scores, primitive_scores)
    class_count = max(2, int(a0_scores.shape[1]))
    posterior = torch.softmax(-a0_scores, dim=1)
    entropy = -(posterior * posterior.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / math.log(float(class_count))
    perturbation = (nominal_candidates - a0_candidates).abs().mean(dim=1)
    return torch.cat((base, entropy.unsqueeze(1), perturbation.unsqueeze(1)), dim=1)


def load_primitive_utility_gate(path_value: str | Path, device: torch.device) -> dict:
    path = repo_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_names = [
        "a0_margin",
        "primitive_margin",
        "top1_agreement",
        "a0_entropy",
        "fusion_perturbation",
    ]
    required = {"feature_names", "mean", "scale", "coef", "intercept"}
    missing = required - set(payload)
    if missing or payload["feature_names"] != expected_names:
        raise ValueError(f"Primitive utility gate {path} has an incompatible feature schema")
    mean = torch.tensor(payload["mean"], device=device, dtype=torch.float32)
    scale = torch.tensor(payload["scale"], device=device, dtype=torch.float32).clamp_min(1e-6)
    coef = torch.tensor(payload["coef"], device=device, dtype=torch.float32)
    if mean.numel() != len(expected_names) or scale.numel() != len(expected_names) or coef.numel() != len(expected_names):
        raise ValueError("Primitive utility gate parameter dimensions are invalid")
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": float(payload["intercept"])}


def supcon_align_loss(
    aux: "C2UAuxHeads",
    hidden: torch.Tensor,
    text_embed: torch.Tensor,
    labels: torch.Tensor,
    seen_labels: list[int],
    tau: float,
    n_text_neg: int,
    device: torch.device,
    topk_hard_bank: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Full-batch Supervised Contrastive alignment: DiT hidden → q_s vs text → q_t.

    Collapse prevention strategy:
    - q_s (skeleton side) is L2-normalised → no class can dominate by magnitude.
    - Negatives include ALL distinct classes in the batch PLUS n_text_neg randomly
      sampled seen-class text embeddings, giving dense coverage of the seen space.
    - When topk_hard_bank is provided, each sample's top-K text-similar
      seen-class negatives (e.g. reading's nearest seen neighbours) are always
      forced into the gallery on top of the random negatives. This is the
      deep-matching analogue of the reconstruction-path topk_hard_neg fix:
      without it, a sample's true hardest text confusions may never appear in
      its negative gallery, so cos(q_s, q_t) is never explicitly pushed apart
      along the direction that matters most at ZSL test time.
    - Temperature tau controls how hard the separation pressure is.
    - This is the training mirror of cosine-argmax inference: the model is
      explicitly trained to maximise cos(q_s_y, q_t_y) over all other q_t_c.
    """
    q_s = aux.project_hidden(hidden)                         # (B, P)
    fc_pos = text_embed[labels]                              # (B, ...) → pooled below
    if fc_pos.dim() == 3:
        fc_pos = fc_pos[:, -1, :]                            # (B, D)
    q_t_pos = aux.project_text(fc_pos)                      # (B, P)

    # Build a text gallery: all unique classes in batch + hard negatives (if
    # available) for every sample's true label + extra random negatives.
    unique_labels = list(dict.fromkeys(labels.tolist()))
    hard_labels: list[int] = []
    if topk_hard_bank is not None:
        for y in labels.tolist():
            hard_labels.extend(int(h) for h in topk_hard_bank[int(y)].tolist())
    combined = list(dict.fromkeys(unique_labels + hard_labels))
    extra_labels = random.choices(seen_labels, k=max(0, n_text_neg - len(combined)))
    gallery_labels = torch.tensor(combined + extra_labels, device=device, dtype=torch.long)
    gallery_embed = text_embed[gallery_labels]               # (G, ...) or (G, D)
    if gallery_embed.dim() == 3:
        gallery_embed = gallery_embed[:, -1, :]             # (G, D)
    q_t_gallery = aux.project_text(gallery_embed)            # (G, P)

    # InfoNCE: each q_s[i] pulls towards q_t_pos[i], repels all gallery texts
    logits = q_s @ q_t_gallery.T / tau                       # (B, G)
    # positive index: position in gallery that matches label
    label_to_gallery_idx = {int(l): i for i, l in enumerate(gallery_labels.tolist())}
    pos_idx = torch.tensor(
        [label_to_gallery_idx.get(int(l), 0) for l in labels.tolist()],
        device=device, dtype=torch.long,
    )
    return F.cross_entropy(logits, pos_idx)

def save_checkpoint(
    path: Path,
    model: DiT,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    metadata: dict,
    aux: C2UAuxHeads | None = None,
    clean_tide: CleanTideNeutralBranch | None = None,
    ema_model: DiT | None = None,
    save_ema_as_model: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = ema_model if save_ema_as_model and ema_model is not None else model
    metadata = dict(metadata)
    if ema_model is not None:
        metadata["use_ema"] = True
        metadata["saved_model"] = "ema" if save_ema_as_model else "raw"
    state = {
        "model": model_to_save.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "aux": aux.state_dict() if aux is not None else None,
        "clean_tide": clean_tide.state_dict() if clean_tide is not None else None,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "metadata": metadata,
        "rng_state": capture_rng_state(),
    }
    # Keep the previous checkpoint readable if a run is interrupted mid-save.
    checkpoint_path = path / "checkpoint.pt"
    checkpoint_temp_path = path / "checkpoint.pt.tmp"
    torch.save(state, checkpoint_temp_path)
    checkpoint_temp_path.replace(checkpoint_path)

    training_state_path = path / "training_state.json"
    training_state_temp_path = path / "training_state.json.tmp"
    training_state_temp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    training_state_temp_path.replace(training_state_path)


def load_checkpoint(
    path: Path,
    model: DiT,
    optimizer=None,
    lr_scheduler=None,
    aux: C2UAuxHeads | None = None,
    clean_tide: CleanTideNeutralBranch | None = None,
    ema_model: DiT | None = None,
) -> dict:
    # Local checkpoints are trusted training artifacts and include optimizer/RNG
    # metadata (including NumPy state) that weights_only=True cannot restore.
    state = torch.load(
        path / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    if ema_model is not None:
        ema_state = state.get("ema_model")
        ema_model.load_state_dict(ema_state if ema_state is not None else state["model"])
    if aux is not None and state.get("aux") is not None:
        aux.load_state_dict(state["aux"])
    if clean_tide is not None and state.get("clean_tide") is not None:
        clean_tide.load_state_dict(state["clean_tide"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    if optimizer is not None:
        restore_rng_state(state.get("rng_state"))
    return state.get("metadata", {})


@torch.no_grad()
def refine_scores_on_frozen_feature_graph(
    scores: torch.Tensor,
    features: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, float]:
    """Confidence-gated label propagation on the frozen evaluation features.

    ``scores`` are lower-is-better A0/C10 energies.  Unlike the existing
    transductive column re-centering, this uses only local geometry in the
    frozen 256-D skeleton-feature space.  The original posterior is injected
    on every iteration, so the graph cannot overwrite A0 with a self-reinforced
    pseudo-label distribution.
    """
    if not bool(config.get("eval_feature_graph", False)) or scores.shape[0] < 2:
        return scores, 0.0

    count, num_classes = scores.shape
    if num_classes < 2:
        return scores, 0.0
    k = min(max(int(config.get("eval_feature_graph_k", 20)), 1), count - 1)
    temperature = max(float(config.get("eval_feature_graph_tau", 0.10)), 1e-6)
    iterations = max(int(config.get("eval_feature_graph_iters", 3)), 1)
    alpha = min(max(float(config.get("eval_feature_graph_alpha", 0.20)), 0.0), 1.0)
    if alpha == 0.0:
        return scores, 0.0

    # Build a directed top-k graph in blocks so NTU120 evaluation does not
    # materialize an N-by-N feature similarity matrix.
    x = F.normalize(features.float().flatten(1), dim=1)
    neighbour_rows: list[torch.Tensor] = []
    neighbour_cols: list[torch.Tensor] = []
    neighbour_values: list[torch.Tensor] = []
    block_size = max(1, int(config.get("eval_feature_graph_block_size", 512)))
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        similarities = x[start:stop] @ x.T
        row_ids = torch.arange(start, stop, device=x.device)
        similarities[torch.arange(stop - start, device=x.device), row_ids] = -torch.inf
        values, columns = torch.topk(similarities, k=k, dim=1)
        neighbour_rows.append(row_ids.unsqueeze(1).expand_as(columns).reshape(-1))
        neighbour_cols.append(columns.reshape(-1))
        neighbour_values.append(values.reshape(-1))

    rows = torch.cat(neighbour_rows)
    cols = torch.cat(neighbour_cols)
    similarities = torch.cat(neighbour_values)
    # Retain only mutual neighbours.  This prevents a large diffuse class from
    # pulling isolated samples through one-way nearest-neighbour edges.
    edge_ids = rows.to(torch.int64) * count + cols.to(torch.int64)
    reverse_ids = cols.to(torch.int64) * count + rows.to(torch.int64)
    sorted_edge_ids = torch.sort(edge_ids).values
    positions = torch.searchsorted(sorted_edge_ids, reverse_ids)
    is_mutual = positions.lt(sorted_edge_ids.numel())
    matched = torch.zeros_like(is_mutual)
    matched[is_mutual] = sorted_edge_ids[positions[is_mutual]].eq(reverse_ids[is_mutual])
    rows, cols, similarities = rows[matched], cols[matched], similarities[matched]
    if rows.numel() == 0:
        return scores, 0.0

    p0 = torch.softmax(-scores, dim=1)
    entropy = -(p0 * p0.clamp_min(1e-8).log()).sum(dim=1) / math.log(float(num_classes))
    confidence = (1.0 - entropy).clamp(0.0, 1.0)
    # Only confident source nodes emit label evidence.  Targets use a separate
    # gate below, leaving confident A0 predictions nearly unchanged.
    edge_values = torch.exp((similarities - 1.0) / temperature) * confidence[cols]
    row_sums = torch.zeros(count, device=scores.device, dtype=edge_values.dtype)
    row_sums.scatter_add_(0, rows, edge_values)
    edge_values = edge_values / row_sums[rows].clamp_min(1e-8)
    posterior = p0
    target_alpha = alpha * (1.0 - confidence)
    for _ in range(iterations):
        # This edge-list reduction is equivalent to sparse graph multiplication
        # but avoids sparse-tensor construction overhead and warnings on PyTorch
        # builds that disable invariant checks globally.
        propagated = torch.zeros_like(posterior)
        propagated.index_add_(0, rows, edge_values.unsqueeze(1) * posterior[cols])
        updated = (1.0 - target_alpha.unsqueeze(1)) * p0 + target_alpha.unsqueeze(1) * propagated
        # Nodes without mutual neighbours retain their original A0 posterior.
        posterior = torch.where(row_sums.unsqueeze(1).gt(0.0), updated, p0)
    return -posterior.clamp_min(1e-8).log(), float(target_alpha.mean().item())


def heteroscedastic_gls_energy_fusion(
    timestep_scores: torch.Tensor,
    timestep_noise_variance: torch.Tensor,
    noise_covariance: torch.Tensor,
    covariance_shrinkage: float,
    ridge: float,
    min_variance_scale: float,
    max_variance_scale: float,
) -> tuple[torch.Tensor, dict]:
    """Fuse correlated timestep energies with label-free heteroscedastic GLS.

    Scores are row-standardized per timestep.  The global correlation matrix
    comes from repeated shared-noise views, rather than score residuals across
    timesteps (which would impose an artificial sum-to-zero negative bias).
    Per-sample noise-view variance rescales that correlation into a covariance
    matrix used for a non-negative GLS weight.
    No labels or tuned class priors enter this transductive estimate.
    """
    if timestep_scores.ndim != 3:
        raise ValueError("D2 timestep scores must have shape (N, T, C)")
    if timestep_noise_variance.shape != timestep_scores.shape[:2]:
        raise ValueError("D2 timestep noise variance must have shape (N, T)")
    sample_count, timestep_count, class_count = timestep_scores.shape
    if timestep_count < 2 or class_count < 2:
        raise ValueError("D2 GLS fusion requires at least two timesteps and classes")
    shrinkage = min(max(float(covariance_shrinkage), 0.0), 1.0)
    ridge = max(float(ridge), 1e-8)
    lower = max(float(min_variance_scale), 1e-4)
    upper = max(float(max_variance_scale), lower)

    if noise_covariance.shape != (timestep_count, timestep_count):
        raise ValueError("D2 noise covariance must have shape (T, T)")
    z = (timestep_scores - timestep_scores.mean(dim=2, keepdim=True)) / timestep_scores.std(
        dim=2, keepdim=True
    ).clamp_min(1e-6)
    covariance = noise_covariance.to(device=z.device, dtype=z.dtype)
    diagonal = covariance.diag().clamp_min(1e-8).sqrt()
    correlation = covariance / diagonal.unsqueeze(1) / diagonal.unsqueeze(0)
    identity = torch.eye(timestep_count, device=z.device, dtype=z.dtype)
    correlation = (1.0 - shrinkage) * correlation + shrinkage * identity
    correlation = 0.5 * (correlation + correlation.T)

    reference_variance = timestep_noise_variance.mean(dim=0, keepdim=True).clamp_min(1e-12)
    relative_variance = (timestep_noise_variance / reference_variance).clamp(lower, upper)
    scale = relative_variance.sqrt()
    covariance_per_sample = (
        scale.unsqueeze(2) * correlation.unsqueeze(0) * scale.unsqueeze(1)
        + ridge * identity.unsqueeze(0)
    )
    ones = torch.ones((sample_count, timestep_count, 1), device=z.device, dtype=z.dtype)
    raw_weights = torch.linalg.solve(covariance_per_sample, ones).squeeze(2)
    negative_fraction = float(raw_weights.lt(0.0).float().mean().item())
    weights = raw_weights.clamp_min(0.0)
    fallback = weights.sum(dim=1, keepdim=True).le(1e-8)
    weights = torch.where(fallback, torch.ones_like(weights), weights)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    fused = (weights.unsqueeze(2) * z).sum(dim=1)
    return fused, {
        "mean_weights": [float(value) for value in weights.mean(dim=0).detach().cpu().tolist()],
        "mean_relative_variance": [
            float(value) for value in relative_variance.mean(dim=0).detach().cpu().tolist()
        ],
        "correlation": correlation.detach().cpu().tolist(),
        "negative_weight_fraction": negative_fraction,
    }


def evaluate_zsl(
    model: DiT,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    text_embed: torch.Tensor,
    unseen_labels: np.ndarray,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    aux: "C2UAuxHeads | None" = None,
    precision: "torch.Tensor | None" = None,
    bridge_ctx: "dict | None" = None,
    analogy_ctx: "dict | None" = None,
    primitive_ctx: "dict | None" = None,
    episodic_proto_ctx: "dict | None" = None,
    clean_tide: "CleanTideNeutralBranch | None" = None,
) -> dict:
    model.eval()
    if aux is not None:
        aux.eval()
    if clean_tide is not None:
        clean_tide.eval()
    prediction_type = str(config.get("prediction_type", "sample"))
    loss_mode = str(config.get("loss_mode", "c2u_feat")).lower()
    eval_distance_space = str(config.get("eval_distance_space", "x0")).lower()
    if eval_distance_space not in {"x0", "sample", "prediction", "native"}:
        raise ValueError(f"Unsupported eval_distance_space: {eval_distance_space}")
    use_native_eval_distance = eval_distance_space in {"prediction", "native"}
    noise_count = int(config.get("eval_num_noise", config.get("num_noise", 1)))
    eval_hetero_multiview = bool(config.get("eval_hetero_multiview", False))
    hetero_metadata: dict | None = None
    eval_seed = config.get("eval_noise_seed", None)
    eval_seed = None if eval_seed is None else int(eval_seed)
    eval_noise_mode = str(config.get("eval_noise_mode", "per_sample")).lower()
    if eval_noise_mode not in {"per_sample", "shared_bank"}:
        raise ValueError(f"Unsupported eval_noise_mode: {eval_noise_mode}")

    def build_shared_noise_bank(count: int, seed_offset: int = 0) -> torch.Tensor:
        generator = None
        if eval_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(eval_seed + seed_offset)
        bank = torch.randn(
            (count, 1, 1, int(config.get("in_channels", 256))),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        return bank.to(device=device, dtype=dtype)

    shared_noise_bank = (
        build_shared_noise_bank(noise_count)
        if eval_noise_mode == "shared_bank"
        else None
    )
    # Mahalanobis scoring: use whitened distance instead of MSE
    use_mahal = bool(config.get("eval_mahal", False)) and precision is not None
    if use_native_eval_distance and use_mahal:
        raise ValueError("Mahalanobis evaluation is only defined for x0 distance space.")
    t_values = config.get("eval_timesteps", None)
    if t_values is None:
        t_values = [int(config["idx_inference_step"])]
    elif isinstance(t_values, int):
        t_values = [int(t_values)]
    else:
        t_values = [int(item) for item in t_values]
    eval_d6_evidence = bool(config.get("eval_d6_evidence", False))
    eval_adaptive_timestep = bool(config.get("eval_adaptive_timestep", False))
    adaptive_metadata: dict | None = None
    if eval_adaptive_timestep and eval_d6_evidence:
        raise ValueError("Adaptive timestep fusion cannot be combined with D6 fixed fusion")
    if eval_adaptive_timestep:
        adaptive_timesteps = config.get("eval_adaptive_timesteps", t_values)
        if isinstance(adaptive_timesteps, int):
            adaptive_timesteps = [int(adaptive_timesteps)]
        else:
            adaptive_timesteps = [int(value) for value in adaptive_timesteps]
        if len(adaptive_timesteps) < 2 or len(set(adaptive_timesteps)) != len(adaptive_timesteps):
            raise ValueError("Adaptive timestep fusion requires at least two unique timesteps")
        if min(adaptive_timesteps) < 0 or max(adaptive_timesteps) >= int(config["num_steps"]):
            raise ValueError("Adaptive timestep fusion contains an invalid timestep")
        t_values = adaptive_timesteps
        noise_count = int(config.get("eval_adaptive_num_noise", noise_count))
        if noise_count < 1:
            raise ValueError("eval_adaptive_num_noise must be positive")
    d6_weights: torch.Tensor | None = None
    d6_metadata: dict | None = None
    if eval_d6_evidence:
        d6_timesteps = config.get("eval_d6_timesteps", t_values)
        if isinstance(d6_timesteps, int):
            d6_timesteps = [int(d6_timesteps)]
        else:
            d6_timesteps = [int(value) for value in d6_timesteps]
        if not d6_timesteps or len(set(d6_timesteps)) != len(d6_timesteps):
            raise ValueError("D6 eval_d6_timesteps must be a non-empty unique timestep list")
        if min(d6_timesteps) < 0 or max(d6_timesteps) >= int(config["num_steps"]):
            raise ValueError("D6 eval_d6_timesteps contains an invalid scheduler timestep")
        weight_values = config.get("eval_d6_weights")
        weight_path = config.get("eval_d6_weight_path")
        if weight_path:
            payload = json.loads(repo_path(weight_path).read_text(encoding="utf-8"))
            payload_timesteps = [int(value) for value in payload.get("timesteps", [])]
            if payload_timesteps != d6_timesteps:
                raise ValueError(
                    "D6 weight file timesteps do not match eval_d6_timesteps: "
                    f"{payload_timesteps} != {d6_timesteps}"
                )
            weight_values = payload.get("weights")
            d6_metadata = payload
        if weight_values is None or len(weight_values) != len(d6_timesteps):
            raise ValueError("D6 requires one frozen non-negative weight per eval_d6_timestep")
        d6_weights = torch.as_tensor(weight_values, device=device, dtype=torch.float32)
        if bool((d6_weights < 0).any()) or float(d6_weights.sum()) <= 0.0:
            raise ValueError("D6 evidence weights must be non-negative with positive sum")
        d6_weights = d6_weights / d6_weights.sum()
        t_values = d6_timesteps
    if eval_hetero_multiview:
        _hetero_t = config.get("eval_hetero_timesteps", [10, 25, 40])
        if isinstance(_hetero_t, int):
            _hetero_t = [int(_hetero_t)]
        else:
            _hetero_t = [int(value) for value in _hetero_t]
        if len(_hetero_t) < 2 or len(set(_hetero_t)) != len(_hetero_t):
            raise ValueError("D2 eval_hetero_timesteps must contain at least two unique timesteps")
        if min(_hetero_t) < 0 or max(_hetero_t) >= int(config["num_steps"]):
            raise ValueError("D2 eval_hetero_timesteps must be valid scheduler timesteps")
        t_values = _hetero_t
        noise_count = int(config.get("eval_hetero_num_noise", noise_count))
        if noise_count < 2:
            raise ValueError("D2 eval_hetero_num_noise must be at least two")
        if bool(config.get("eval_mahal", False)) or float(config.get("eval_cfg_scale", 1.0)) != 1.0:
            raise ValueError("D2 multiview fusion currently requires MSE/x0 scores and eval_cfg_scale=1")
    if eval_d6_evidence and eval_hetero_multiview:
        raise ValueError("D6 evidence integration cannot be combined with D2 hetero multiview")
    reverse_x0_timesteps = config.get("eval_reverse_x0_timesteps", None)
    if reverse_x0_timesteps is not None:
        if isinstance(reverse_x0_timesteps, int):
            reverse_x0_timesteps = [int(reverse_x0_timesteps)]
        else:
            reverse_x0_timesteps = [int(value) for value in reverse_x0_timesteps]
        if not reverse_x0_timesteps:
            raise ValueError("eval_reverse_x0_timesteps must be non-empty")
        if min(reverse_x0_timesteps) < 1 or max(reverse_x0_timesteps) >= int(config["num_steps"]):
            raise ValueError("eval_reverse_x0_timesteps must stay within [1, num_steps - 1]")
        if any(
            current <= following
            for current, following in zip(reverse_x0_timesteps, reverse_x0_timesteps[1:])
        ):
            raise ValueError("eval_reverse_x0_timesteps must be strictly descending")
        if len(t_values) != 1 or int(t_values[0]) != reverse_x0_timesteps[0]:
            raise ValueError(
                "E1 requires exactly one eval timestep equal to the first eval_reverse_x0_timesteps entry"
            )
        if use_native_eval_distance:
            raise ValueError("E1 reverse x0 evaluation requires eval_distance_space=x0")
    label_to_idx = {int(label): idx for idx, label in enumerate(unseen_labels)}
    score_chunks: list[torch.Tensor] = []
    d6_score_chunks: list[torch.Tensor] = []
    adaptive_score_chunks: list[torch.Tensor] = []
    adaptive_variance_chunks: list[torch.Tensor] = []
    hetero_score_chunks: list[torch.Tensor] = []
    hetero_noise_variance_chunks: list[torch.Tensor] = []
    hetero_noise_covariance_numerators: list[torch.Tensor] = []
    hetero_noise_covariance_counts: list[int] = []
    label_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    start_time = time.time()

    eval_align_mode = loss_mode == "c2u_feat_align" and aux is not None
    eval_also_proj_score = (
        bool(config.get("eval_also_proj_score", False))
        and aux is not None
        and not eval_align_mode
    )
    eval_primitive_score = (
        bool(config.get("eval_primitive_score", False))
        and aux is not None
        and primitive_ctx is not None
        and not eval_align_mode
    )
    eval_clean_tide_multilayer = (
        loss_mode == "tdsm_x0_action_primitive_clean_tide_multilayer"
        and clean_tide is not None
    )
    eval_clean_tide_primitive_score = (
        bool(config.get("eval_clean_tide_primitive_score", False))
        and clean_tide is not None
        and aux is not None
        and primitive_ctx is not None
        and not eval_align_mode
    )
    eval_feature_distribution_score = (
        bool(config.get("eval_primitive_feature_distribution_score", False))
        and aux is not None
        and primitive_ctx is not None
        and not eval_align_mode
    )
    eval_episodic_proto_score = (
        bool(config.get("eval_episodic_proto_score", False))
        and aux is not None
        and primitive_ctx is not None
        and episodic_proto_ctx is not None
        and aux.episodic_proto_generator is not None
        and not eval_align_mode
    )
    eval_f1_gallery_match = (
        bool(config.get("eval_f1_gallery_match", False))
        and aux is not None
        and aux.frozen_gallery_feature_proj is not None
        and not eval_align_mode
    )

    # Pre-compute projection-space ingredients outside the loop (no bsz yet).
    neutral_fc_proj: torch.Tensor | None = None
    q_t_proj: list[torch.Tensor] = []
    proj_t_values: list[int] = []
    proj_noise_count: int = 1
    proj_score_chunks: list[torch.Tensor] = []
    primitive_t_values: list[int] = []
    primitive_noise_count: int = 1
    primitive_score_chunks: list[torch.Tensor] = []
    clean_tide_primitive_score_chunks: list[torch.Tensor] = []
    feature_distribution_score_chunks: list[torch.Tensor] = []
    feature_distribution_mean: torch.Tensor | None = None
    feature_distribution_log_variance: torch.Tensor | None = None
    episodic_proto_score_chunks: list[torch.Tensor] = []
    episodic_proto_mean: torch.Tensor | None = None
    episodic_proto_reliability = 0.0
    episodic_proto_seen_accuracy = 0.0
    f1_gallery_match_score_chunks: list[torch.Tensor] = []
    f1_unseen_text: torch.Tensor | None = None
    mahal_score_chunks: list[torch.Tensor] = []
    bridge_score_chunks: list[torch.Tensor] = []
    analogy_score_chunks: list[torch.Tensor] = []
    collect_both: bool = False  # set per-batch inside the loop, initialized here for scope

    if eval_f1_gallery_match:
        f1_unseen_text = pooled_text_embed(text_embed)[
            torch.as_tensor(unseen_labels, device=device, dtype=torch.long)
        ]

    if eval_also_proj_score:
        _proj_t = config.get("eval_proj_timesteps", None)
        if _proj_t is None:
            _proj_t = [t_values[len(t_values) // 2]]
        elif isinstance(_proj_t, int):
            _proj_t = [int(_proj_t)]
        else:
            _proj_t = [int(v) for v in _proj_t]
        proj_t_values = _proj_t
        proj_noise_count = int(config.get("eval_proj_num_noise", 1))
        # q_t for all unseen classes — text side, no DiT needed
        for _ulabel in unseen_labels:
            _lt = torch.full((1,), int(_ulabel), device=device, dtype=torch.long)
            _fc_t, _ = split_text_condition(text_embed, _lt)
            q_t_proj.append(aux.project_text(_fc_t))  # (1, P)

    if eval_primitive_score:
        _primitive_t = config.get("eval_primitive_timesteps", None)
        if _primitive_t is None:
            _primitive_t = [t_values[len(t_values) // 2]]
        elif isinstance(_primitive_t, int):
            _primitive_t = [int(_primitive_t)]
        else:
            _primitive_t = [int(value) for value in _primitive_t]
        primitive_t_values = _primitive_t
        primitive_noise_count = int(config.get("eval_primitive_num_noise", 1))

    if eval_feature_distribution_score:
        candidate_targets = primitive_ctx["class_targets"][
            torch.as_tensor(unseen_labels, device=device, dtype=torch.long)
        ].to(dtype=dtype)
        feature_distribution_mean, feature_distribution_log_variance = (
            aux.primitive_feature_distribution_parameters(candidate_targets)
        )
    if eval_episodic_proto_score:
        candidate_targets = primitive_ctx["class_targets"][
            torch.as_tensor(unseen_labels, device=device, dtype=torch.long)
        ].to(dtype=dtype)
        episodic_proto_mean = aux.episodic_proto_generator(
            candidate_targets,
            primitive_ctx["class_targets"][episodic_proto_ctx["seen_labels"]].to(dtype=dtype),
            episodic_proto_ctx["prototypes"].to(dtype=dtype),
            heldout_indices=None,
            temperature=float(config.get("episodic_proto_temperature", 0.2)),
        )
        episodic_proto_seen_accuracy, episodic_proto_reliability = episodic_proto_seen_reliability(
            aux,
            primitive_ctx,
            episodic_proto_ctx,
            temperature=float(config.get("episodic_proto_temperature", 0.2)),
        )

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(loader, desc="zsl eval")):
            features = features.to(device=device, dtype=dtype).unsqueeze(1)
            labels = labels.to(device=device, dtype=torch.long)
            if bool(config.get("eval_feature_graph", False)):
                feature_chunks.append(features.detach().float().flatten(1))
            bsz = features.shape[0]
            clean_token_residuals = None
            if eval_clean_tide_multilayer:
                clean_token_residuals = clean_tide.condition_residuals(clean_tide.encode(features))
            scores = torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
            proj_scores = torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
            if eval_f1_gallery_match:
                if f1_unseen_text is None:
                    raise RuntimeError("F1 unseen text gallery was not initialized")
                f1_logits, _f1_gate = aux.frozen_gallery_match_logits(
                    features.flatten(1),
                    f1_unseen_text,
                )
                # Evaluation scores use lower-is-better convention throughout.
                f1_gallery_match_score_chunks.append((-f1_logits).detach().float())
            if eval_also_proj_score and neutral_fc_proj is None:
                # Store as (1, ...) — expand per-batch to handle variable last-batch size.
                neutral_mode = str(config.get("eval_proj_neutral_mode", "mean")).lower()
                if neutral_mode == "zero":
                    neutral_fc_proj = torch.zeros_like(pooled_text_embed(text_embed)[:1])
                elif neutral_mode == "mean":
                    neutral_fc_proj = pooled_text_embed(text_embed).mean(0, keepdim=True)
                else:
                    raise ValueError(f"Unsupported eval_proj_neutral_mode: {neutral_mode}")
                q_t_proj = [q for q in q_t_proj]  # already (1, P) from pre-compute

            if bridge_ctx is not None:
                bridge_score_chunks.append(compute_bridge_scores(features, bridge_ctx).detach().float())

            if analogy_ctx is not None:
                analogy_score_chunks.append(compute_seen_analogy_scores(features, analogy_ctx).detach().float())

            if eval_align_mode:
                # Clean cosine scoring: neutral-conditioned anchor → Proj_s(q_s)
                # vs all candidate texts → Proj_t(q_t_c).
                #
                # The anchor DiT hidden state is extracted using the MEAN of all
                # all class text embeddings as a class-neutral condition.  This
                # prevents any class label (true or candidate) from influencing
                # q_s, so the score is purely skeleton→text similarity.
                #
                # Per-candidate conditioning (prior version) was self-referential:
                # cos(Proj_s(hidden_c), Proj_t(text_c)) measures self-consistency
                # of text_c, not how well the skeleton matches text_c.
                #
                # True-label conditioning (first version) was an outright data leak.
                align_t_values = config.get("eval_align_timesteps", None)
                if align_t_values is None:
                    align_t_values = [int(t_values[len(t_values) // 2])]
                elif isinstance(align_t_values, int):
                    align_t_values = [int(align_t_values)]
                else:
                    align_t_values = [int(item) for item in align_t_values]
                align_noise_count = int(config.get("eval_align_num_noise", 1))
                denom = max(1, align_noise_count * len(align_t_values))
                align_shared_noise_bank = (
                    build_shared_noise_bank(align_noise_count, seed_offset=10000)
                    if eval_noise_mode == "shared_bank"
                    else None
                )

                # Neutral condition: mean pooled text across all classes.
                neutral_fc = pooled_text_embed(text_embed).mean(0, keepdim=True).expand(bsz, -1)
                # Pre-compute q_t for all unseen classes (text side, no model needed)
                q_t_all = []
                for label in unseen_labels:
                    lt = torch.full((bsz,), int(label), device=device, dtype=torch.long)
                    fc_t, _ = split_text_condition(text_embed, lt)
                    q_t_all.append(aux.project_text(fc_t))  # (B, P)

                for t_idx, t_val in enumerate(align_t_values):
                    t_float = torch.ones((bsz,), device=device, dtype=dtype) * int(t_val)
                    t_long = t_float.long()
                    for noise_idx in range(align_noise_count):
                        if align_shared_noise_bank is not None:
                            noise = align_shared_noise_bank[noise_idx].expand(bsz, -1, -1)
                        elif eval_seed is None:
                            noise = torch.randn_like(features)
                        else:
                            noise_seed = eval_seed + batch_idx * 1000003 + t_idx * 1009 + noise_idx
                            generator = torch.Generator(device=device).manual_seed(noise_seed)
                            noise = torch.randn(features.shape, device=device, dtype=dtype, generator=generator)
                        noisy = scheduler.add_noise(features, noise, t_long)
                        _, hidden_states = model(
                            noisy, t_float, neutral_fc, None,
                            return_hidden=True,
                            hidden_layers=[int(config.get("align_layer", config.get("depth", 8)))],
                        )
                        q_s = aux.project_hidden(DiT.pool_hidden(hidden_states[0]))  # (B, P)
                        for col, q_t in enumerate(q_t_all):
                            scores[:, col] += -(q_s * q_t).sum(dim=1)
                scores = scores / denom
            else:
                cfg_scale = float(config.get("eval_cfg_scale", 1.0))
                mahal_alpha = float(config.get("eval_mahal_alpha", 1.0))  # 1.0=pure Mahal, 0.0=pure MSE
                # When alpha is strictly between 0 and 1, collect both score types.
                collect_both = use_mahal and 0.0 < mahal_alpha < 1.0
                mahal_scores = torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32) if collect_both else None
                # Null condition: mean of all class pooled embeddings.
                null_fc_cfg = pooled_text_embed(text_embed).mean(0, keepdim=True).expand(bsz, -1)
                hetero_timestep_scores: list[torch.Tensor] = []
                hetero_timestep_noise_variances: list[torch.Tensor] = []
                hetero_timestep_noise_score_views: list[torch.Tensor] = []
                d6_timestep_scores: list[torch.Tensor] = []
                adaptive_timestep_scores: list[torch.Tensor] = []
                adaptive_timestep_variances: list[torch.Tensor] = []

                for t_idx, t_value in enumerate(t_values):
                    t_float = torch.ones((bsz,), device=device, dtype=dtype) * int(t_value)
                    t_long = t_float.long()
                    hetero_noise_scores: list[torch.Tensor] = []
                    d6_timestep_score = (
                        torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
                        if (eval_d6_evidence or eval_adaptive_timestep)
                        else None
                    )
                    adaptive_noise_score = (
                        torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
                        if eval_adaptive_timestep
                        else None
                    )
                    adaptive_noise_views: list[torch.Tensor] = []
                    for noise_idx in range(noise_count):
                        hetero_noise_score = (
                            torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
                            if eval_hetero_multiview
                            else None
                        )
                        # Keep an independent per-noise view for D9 stability.
                        # The accumulated score remains the mean-energy numerator,
                        # but must not be reused as a variance sample.
                        adaptive_noise_view = (
                            torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
                            if eval_adaptive_timestep
                            else None
                        )
                        if shared_noise_bank is not None:
                            noise = shared_noise_bank[noise_idx].expand(bsz, -1, -1)
                        elif eval_seed is None:
                            noise = torch.randn_like(features)
                        else:
                            noise_seed = (
                                eval_seed + batch_idx * 1000003 + noise_idx + 40000
                                if eval_hetero_multiview
                                else eval_seed + batch_idx * 1000003 + t_idx * 1009 + noise_idx
                            )
                            generator = torch.Generator(device=device).manual_seed(noise_seed)
                            noise = torch.randn(features.shape, device=device, dtype=dtype, generator=generator)
                        noisy = scheduler.add_noise(features, noise, t_long)
                        native_target = (
                            diffusion_training_target(
                                prediction_type,
                                features,
                                noise,
                                t_long,
                                scheduler,
                            )
                            if use_native_eval_distance
                            else None
                        )

                        if cfg_scale > 1.0 and reverse_x0_timesteps is None:
                            pred_null = model(
                                noisy,
                                t_float,
                                null_fc_cfg,
                                None,
                                clean_token_residuals=clean_token_residuals,
                            )

                        for col, label in enumerate(unseen_labels):
                            label_tensor = torch.full((bsz,), int(label), device=device, dtype=torch.long)
                            fc, fl = split_text_condition(text_embed, label_tensor)
                            if use_native_eval_distance:
                                pred = model(
                                    noisy,
                                    t_float,
                                    fc,
                                    fl,
                                    clean_token_residuals=clean_token_residuals,
                                )
                                if cfg_scale > 1.0:
                                    pred = (1.0 + cfg_scale) * pred - cfg_scale * pred_null
                                pred_sample = None
                                mse_dist = reconstruction_distance(pred, native_target)
                            else:
                                if reverse_x0_timesteps is None:
                                    pred = model(
                                        noisy,
                                        t_float,
                                        fc,
                                        fl,
                                        clean_token_residuals=clean_token_residuals,
                                    )
                                    if cfg_scale > 1.0:
                                        pred = (1.0 + cfg_scale) * pred - cfg_scale * pred_null
                                    pred_sample = prediction_to_sample(
                                        pred,
                                        noisy,
                                        t_long,
                                        scheduler,
                                        prediction_type,
                                    )
                                else:
                                    pred_sample = reverse_x0_deterministic(
                                        model=model,
                                        noisy=noisy,
                                        reverse_timesteps=reverse_x0_timesteps,
                                        fc=fc,
                                        fl=fl,
                                        scheduler=scheduler,
                                        prediction_type=prediction_type,
                                        dtype=dtype,
                                        cfg_scale=cfg_scale,
                                        null_fc=null_fc_cfg,
                                        clean_token_residuals=clean_token_residuals,
                                    )
                                mse_dist = reconstruction_distance(pred_sample, features)
                            if d6_timestep_score is not None:
                                d6_timestep_score[:, col] += mse_dist
                            if adaptive_noise_score is not None:
                                adaptive_noise_score[:, col] += mse_dist
                                adaptive_noise_view[:, col] += mse_dist
                            elif collect_both:
                                scores[:, col] += mse_dist
                                mahal_scores[:, col] += whitened_distance(pred_sample, features, precision)
                            elif use_mahal:
                                scores[:, col] += whitened_distance(pred_sample, features, precision)
                            elif eval_hetero_multiview:
                                hetero_noise_score[:, col] += mse_dist
                            else:
                                scores[:, col] += mse_dist
                        if hetero_noise_score is not None:
                            hetero_noise_scores.append(hetero_noise_score)
                        if adaptive_noise_score is not None:
                            adaptive_noise_views.append(adaptive_noise_view)

                    if eval_hetero_multiview:
                        stacked_noise_scores = torch.stack(hetero_noise_scores, dim=0)
                        hetero_timestep_scores.append(stacked_noise_scores.mean(dim=0))
                        hetero_timestep_noise_score_views.append(stacked_noise_scores)
                        hetero_timestep_noise_variances.append(
                            stacked_noise_scores.var(dim=0, unbiased=False).mean(dim=1)
                        )
                    if d6_timestep_score is not None:
                        mean_timestep_score = d6_timestep_score / max(1, noise_count)
                        if eval_d6_evidence:
                            d6_timestep_scores.append(mean_timestep_score)
                        if eval_adaptive_timestep:
                            adaptive_timestep_scores.append(mean_timestep_score)
                            adaptive_timestep_variances.append(
                                torch.stack(adaptive_noise_views, dim=0)
                                .var(dim=0, unbiased=False)
                                .mean(dim=1)
                            )

                denom_s = max(1, noise_count * len(t_values))
                if eval_d6_evidence:
                    d6_score_chunks.append(torch.stack(d6_timestep_scores, dim=1).detach().float())
                elif eval_adaptive_timestep:
                    adaptive_score_chunks.append(
                        torch.stack(adaptive_timestep_scores, dim=1).detach().float()
                    )
                    adaptive_variance_chunks.append(
                        torch.stack(adaptive_timestep_variances, dim=1).detach().float()
                    )
                elif eval_hetero_multiview:
                    hetero_batch_scores = torch.stack(hetero_timestep_scores, dim=1)
                    hetero_score_chunks.append(hetero_batch_scores.detach().float())
                    hetero_noise_variance_chunks.append(
                        torch.stack(hetero_timestep_noise_variances, dim=1).detach().float()
                    )
                    noise_view_tensor = torch.stack(hetero_timestep_noise_score_views, dim=2)
                    noise_residual = noise_view_tensor - noise_view_tensor.mean(dim=0, keepdim=True)
                    hetero_noise_covariance_numerators.append(
                        torch.einsum("mbtc,mbuc->tu", noise_residual, noise_residual).detach().float()
                    )
                    hetero_noise_covariance_counts.append(
                        int(noise_view_tensor.shape[0] * noise_view_tensor.shape[1] * noise_view_tensor.shape[3])
                    )
                    # Retain an arithmetic mean as a defensive fallback until
                    # global GLS covariance is estimated after all test rows.
                    scores = hetero_batch_scores.mean(dim=1)
                else:
                    scores = scores / denom_s
                if collect_both:
                    mahal_scores = mahal_scores / denom_s

                # Projection-space secondary scoring (eval_also_proj_score: true).
                # Uses neutral-conditioned hidden state → Proj_s(q_s) vs Proj_t(q_t_c).
                if eval_also_proj_score:
                    proj_shared_noise_bank = (
                        build_shared_noise_bank(proj_noise_count, seed_offset=20000)
                        if eval_noise_mode == "shared_bank"
                        else None
                    )
                    for t_idx2, t_value2 in enumerate(proj_t_values):
                        t_float2 = torch.ones((bsz,), device=device, dtype=dtype) * int(t_value2)
                        t_long2 = t_float2.long()
                        for noise_idx2 in range(proj_noise_count):
                            if proj_shared_noise_bank is not None:
                                noise2 = proj_shared_noise_bank[noise_idx2].expand(bsz, -1, -1)
                            elif eval_seed is None:
                                noise2 = torch.randn_like(features)
                            else:
                                ns2 = eval_seed + batch_idx * 1000003 + t_idx2 * 1009 + noise_idx2 + 5000
                                gen2 = torch.Generator(device=device).manual_seed(ns2)
                                noise2 = torch.randn(features.shape, device=device, dtype=dtype, generator=gen2)
                            noisy2 = scheduler.add_noise(features, noise2, t_long2)
                            _neutral_exp = neutral_fc_proj.expand(bsz, -1)
                            _, hidden_states2 = model(
                                noisy2, t_float2, _neutral_exp, None,
                                return_hidden=True,
                                hidden_layers=[int(config.get("align_layer", config.get("depth", 8)))],
                            )
                            q_s2 = aux.project_hidden(DiT.pool_hidden(hidden_states2[0]))
                            for col2, q_t2 in enumerate(q_t_proj):
                                proj_scores[:, col2] += -(q_s2 * q_t2.expand(bsz, -1)).sum(dim=1)
                    proj_scores = proj_scores / max(1, proj_noise_count * len(proj_t_values))
                    proj_score_chunks.append(proj_scores.detach().float())

                if eval_primitive_score:
                    primitive_logits = torch.zeros(
                        (bsz, int(primitive_ctx["prototypes"].shape[0])),
                        device=device,
                        dtype=torch.float32,
                    )
                    primitive_shared_noise_bank = (
                        build_shared_noise_bank(primitive_noise_count, seed_offset=30000)
                        if eval_noise_mode == "shared_bank"
                        else None
                    )
                    for primitive_t_idx, primitive_t_value in enumerate(primitive_t_values):
                        primitive_t_float = torch.full(
                            (bsz,),
                            int(primitive_t_value),
                            device=device,
                            dtype=dtype,
                        )
                        primitive_t_long = primitive_t_float.long()
                        for primitive_noise_idx in range(primitive_noise_count):
                            if primitive_shared_noise_bank is not None:
                                primitive_noise = primitive_shared_noise_bank[
                                    primitive_noise_idx
                                ].expand(bsz, -1, -1)
                            elif eval_seed is None:
                                primitive_noise = torch.randn_like(features)
                            else:
                                primitive_seed = (
                                    eval_seed
                                    + batch_idx * 1000003
                                    + primitive_t_idx * 1009
                                    + primitive_noise_idx
                                    + 15000
                                )
                                primitive_generator = torch.Generator(device=device).manual_seed(
                                    primitive_seed
                                )
                                primitive_noise = torch.randn(
                                    features.shape,
                                    device=device,
                                    dtype=dtype,
                                    generator=primitive_generator,
                                )
                            primitive_noisy = scheduler.add_noise(
                                features,
                                primitive_noise,
                                primitive_t_long,
                            )
                            primitive_neutral_fc = torch.zeros(
                                (bsz, int(text_embed.shape[-1])),
                                device=device,
                                dtype=dtype,
                            )
                            _, primitive_hidden_states = model(
                                primitive_noisy,
                                primitive_t_float,
                                primitive_neutral_fc,
                                None,
                                return_hidden=True,
                                hidden_layers=[int(config.get("align_layer", 8))],
                                clean_token_residuals=clean_token_residuals,
                            )
                            primitive_logits += action_primitive_logits(
                                aux,
                                primitive_hidden_states[0],
                                primitive_ctx,
                                logit_scale=float(config.get("primitive_logit_scale", 10.0)),
                                logit_bias=float(config.get("primitive_logit_bias", -2.0)),
                            )
                    primitive_logits = primitive_logits / max(
                        1,
                        primitive_noise_count * len(primitive_t_values),
                    )
                    primitive_scores = action_primitive_class_scores(
                        primitive_logits,
                        unseen_labels,
                        primitive_ctx,
                        negative_weight=float(
                            config.get("eval_primitive_negative_weight", 0.1)
                        ),
                        score_mode=str(config.get("eval_primitive_score_mode", "c10")),
                        pu_positive_probability=float(
                            config.get("primitive_pu_positive_probability", 0.95)
                        ),
                    )
                    primitive_score_chunks.append(primitive_scores.detach().float())

                if eval_clean_tide_primitive_score:
                    clean_tide_logits = action_primitive_logits(
                        aux,
                        clean_tide.encode(features),
                        primitive_ctx,
                        logit_scale=float(config.get("primitive_logit_scale", 10.0)),
                        logit_bias=float(config.get("primitive_logit_bias", -2.0)),
                    )
                    clean_tide_scores = action_primitive_class_scores(
                        clean_tide_logits,
                        unseen_labels,
                        primitive_ctx,
                        negative_weight=float(config.get("eval_primitive_negative_weight", 0.1)),
                        score_mode=str(config.get("eval_primitive_score_mode", "c10")),
                        pu_positive_probability=float(
                            config.get("primitive_pu_positive_probability", 0.95)
                        ),
                    )
                    clean_tide_primitive_score_chunks.append(clean_tide_scores.detach().float())

                if (
                    eval_feature_distribution_score
                    and feature_distribution_mean is not None
                    and feature_distribution_log_variance is not None
                ):
                    feature_target = features.flatten(1).float().unsqueeze(1)
                    feature_mean = feature_distribution_mean.float().unsqueeze(0)
                    feature_log_variance = feature_distribution_log_variance.float().unsqueeze(0)
                    feature_distribution_scores = 0.5 * (
                        (feature_target - feature_mean).square()
                        * torch.exp(-feature_log_variance)
                        + feature_log_variance
                    ).mean(dim=-1)
                    feature_distribution_score_chunks.append(
                        feature_distribution_scores.detach().float()
                    )

                if eval_episodic_proto_score and episodic_proto_mean is not None:
                    proto_target = features.flatten(1).float().unsqueeze(1)
                    proto_variance = episodic_proto_ctx["shared_variance"].float().view(1, 1, -1)
                    proto_scores = 0.5 * (
                        (proto_target - episodic_proto_mean.float().unsqueeze(0)).square()
                        / proto_variance
                        + proto_variance.log()
                    ).mean(dim=-1)
                    episodic_proto_score_chunks.append(proto_scores.detach().float())

            score_chunks.append(scores.detach().float())
            label_chunks.append(labels.detach())
            if collect_both and mahal_scores is not None:
                mahal_score_chunks.append(mahal_scores.detach().float())

    if score_chunks:
        all_scores = torch.cat(score_chunks, dim=0)
        all_labels = torch.cat(label_chunks, dim=0)
    else:
        all_scores = torch.empty((0, len(unseen_labels)), device=device, dtype=torch.float32)
        all_labels = torch.empty((0,), device=device, dtype=torch.long)

    # D6 combines calibrated conditional evidence, not raw MSEs. Each
    # timestep gets its own label-free column bias removal and row scale before
    # the frozen seen-only weights are applied.
    if eval_d6_evidence:
        if d6_weights is None or not d6_score_chunks:
            raise RuntimeError("D6 did not collect per-timestep conditional energies")
        all_d6_scores = torch.cat(d6_score_chunks, dim=0)
        calibrated_views: list[torch.Tensor] = []
        for timestep_index in range(all_d6_scores.shape[1]):
            view = all_d6_scores[:, timestep_index, :]
            if bool(config.get("eval_d6_per_timestep_a3", True)):
                view = view - view.mean(dim=0, keepdim=True)
            if bool(config.get("eval_d6_per_timestep_row_zscore", True)):
                view = (view - view.mean(dim=1, keepdim=True)) / view.std(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
            calibrated_views.append(view)
        all_scores = torch.stack(calibrated_views, dim=1).mul(
            d6_weights.view(1, -1, 1)
        ).sum(dim=1)

    if eval_adaptive_timestep:
        if not adaptive_score_chunks or not adaptive_variance_chunks:
            raise RuntimeError("Adaptive timestep fusion did not collect per-timestep scores")
        all_adaptive_scores = torch.cat(adaptive_score_chunks, dim=0)
        all_adaptive_variances = torch.cat(adaptive_variance_chunks, dim=0)
        calibrated_adaptive_views: list[torch.Tensor] = []
        for timestep_index in range(all_adaptive_scores.shape[1]):
            view = all_adaptive_scores[:, timestep_index, :]
            if bool(config.get("eval_adaptive_per_timestep_a3", True)):
                view = view - view.mean(dim=0, keepdim=True)
            if bool(config.get("eval_adaptive_per_timestep_row_zscore", True)):
                view = (view - view.mean(dim=1, keepdim=True)) / view.std(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
            calibrated_adaptive_views.append(view)
        calibrated_tensor = torch.stack(calibrated_adaptive_views, dim=1)
        sorted_scores = calibrated_tensor.sort(dim=2).values
        margin = (sorted_scores[:, :, 1] - sorted_scores[:, :, 0]).clamp_min(0.0)
        variance_penalty = all_adaptive_variances.sqrt().clamp_min(0.0)
        evidence = margin / (
            variance_penalty
            + float(config.get("eval_adaptive_variance_floor", 0.05))
        )
        temperature = max(float(config.get("eval_adaptive_temperature", 0.50)), 1e-6)
        weights = torch.softmax(evidence / temperature, dim=1)
        anchor = int(config.get("eval_adaptive_anchor_timestep", 25))
        anchor_index = min(
            range(len(t_values)),
            key=lambda index: abs(int(t_values[index]) - anchor),
        )
        anchor_floor = min(
            max(float(config.get("eval_adaptive_anchor_floor", 0.20)), 0.0),
            1.0,
        )
        if anchor_floor > 0.0:
            anchor_one_hot = torch.zeros_like(weights)
            anchor_one_hot[:, anchor_index] = 1.0
            weights = (1.0 - anchor_floor) * weights + anchor_floor * anchor_one_hot
        all_scores = (calibrated_tensor * weights.unsqueeze(-1)).sum(dim=1)
        adaptive_metadata = {
            "timesteps": [int(value) for value in t_values],
            "mean_weights": weights.mean(dim=0).detach().cpu().tolist(),
            "mean_evidence": evidence.mean(dim=0).detach().cpu().tolist(),
            "mean_margin": margin.mean(dim=0).detach().cpu().tolist(),
            "mean_variance_penalty": variance_penalty.mean(dim=0).detach().cpu().tolist(),
            "weight_entropy": float(
                (-(weights * weights.clamp_min(1e-8).log()).sum(dim=1)).mean().item()
            ),
        }

    if eval_hetero_multiview:
        if (
            not hetero_score_chunks
            or not hetero_noise_variance_chunks
            or not hetero_noise_covariance_numerators
        ):
            raise RuntimeError("D2 multiview evaluation did not collect timestep scores")
        all_hetero_scores = torch.cat(hetero_score_chunks, dim=0)
        all_hetero_variances = torch.cat(hetero_noise_variance_chunks, dim=0)
        all_hetero_noise_covariance = torch.stack(hetero_noise_covariance_numerators).sum(dim=0) / float(
            max(1, sum(hetero_noise_covariance_counts))
        )
        all_scores, hetero_metadata = heteroscedastic_gls_energy_fusion(
            all_hetero_scores,
            all_hetero_variances,
            all_hetero_noise_covariance,
            covariance_shrinkage=float(config.get("eval_hetero_covariance_shrinkage", 0.10)),
            ridge=float(config.get("eval_hetero_ridge", 0.05)),
            min_variance_scale=float(config.get("eval_hetero_min_variance_scale", 0.25)),
            max_variance_scale=float(config.get("eval_hetero_max_variance_scale", 4.0)),
        )

    all_proj_scores: torch.Tensor | None = None
    if eval_also_proj_score and proj_score_chunks:
        all_proj_scores = torch.cat(proj_score_chunks, dim=0)
        if bool(config.get("eval_proj_a3_calib", False)):
            all_proj_scores = all_proj_scores - all_proj_scores.mean(dim=0, keepdim=True)
        if bool(config.get("eval_proj_row_zscore", config.get("eval_row_zscore", True))):
            all_proj_scores = (
                all_proj_scores - all_proj_scores.mean(dim=1, keepdim=True)
            ) / all_proj_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    all_primitive_scores: torch.Tensor | None = None
    if eval_primitive_score and primitive_score_chunks:
        all_primitive_scores = torch.cat(primitive_score_chunks, dim=0)
        if bool(config.get("eval_primitive_a3_calib", True)):
            all_primitive_scores = all_primitive_scores - all_primitive_scores.mean(
                dim=0,
                keepdim=True,
            )
        if bool(config.get("eval_primitive_row_zscore", True)):
            all_primitive_scores = (
                all_primitive_scores - all_primitive_scores.mean(dim=1, keepdim=True)
            ) / all_primitive_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    all_clean_tide_primitive_scores: torch.Tensor | None = None
    if eval_clean_tide_primitive_score and clean_tide_primitive_score_chunks:
        all_clean_tide_primitive_scores = torch.cat(clean_tide_primitive_score_chunks, dim=0)
        if bool(config.get("eval_clean_tide_primitive_a3_calib", True)):
            all_clean_tide_primitive_scores = (
                all_clean_tide_primitive_scores
                - all_clean_tide_primitive_scores.mean(dim=0, keepdim=True)
            )
        if bool(config.get("eval_clean_tide_primitive_row_zscore", True)):
            all_clean_tide_primitive_scores = (
                all_clean_tide_primitive_scores
                - all_clean_tide_primitive_scores.mean(dim=1, keepdim=True)
            ) / all_clean_tide_primitive_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    all_feature_distribution_scores: torch.Tensor | None = None
    if eval_feature_distribution_score and feature_distribution_score_chunks:
        all_feature_distribution_scores = torch.cat(feature_distribution_score_chunks, dim=0)
        if bool(config.get("eval_primitive_feature_distribution_a3_calib", True)):
            all_feature_distribution_scores = (
                all_feature_distribution_scores
                - all_feature_distribution_scores.mean(dim=0, keepdim=True)
            )
        if bool(config.get("eval_primitive_feature_distribution_row_zscore", True)):
            all_feature_distribution_scores = (
                all_feature_distribution_scores
                - all_feature_distribution_scores.mean(dim=1, keepdim=True)
            ) / all_feature_distribution_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    all_episodic_proto_scores: torch.Tensor | None = None
    if eval_episodic_proto_score and episodic_proto_score_chunks:
        all_episodic_proto_scores = torch.cat(episodic_proto_score_chunks, dim=0)
        if bool(config.get("eval_episodic_proto_a3_calib", True)):
            all_episodic_proto_scores = (
                all_episodic_proto_scores
                - all_episodic_proto_scores.mean(dim=0, keepdim=True)
            )
        if bool(config.get("eval_episodic_proto_row_zscore", True)):
            all_episodic_proto_scores = (
                all_episodic_proto_scores
                - all_episodic_proto_scores.mean(dim=1, keepdim=True)
            ) / all_episodic_proto_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    all_f1_gallery_match_scores: torch.Tensor | None = None
    if eval_f1_gallery_match and f1_gallery_match_score_chunks:
        all_f1_gallery_match_scores = torch.cat(f1_gallery_match_score_chunks, dim=0)
        if bool(config.get("eval_f1_gallery_match_a3_calib", True)):
            all_f1_gallery_match_scores = (
                all_f1_gallery_match_scores
                - all_f1_gallery_match_scores.mean(dim=0, keepdim=True)
            )
        if bool(config.get("eval_f1_gallery_match_row_zscore", True)):
            all_f1_gallery_match_scores = (
                all_f1_gallery_match_scores
                - all_f1_gallery_match_scores.mean(dim=1, keepdim=True)
            ) / all_f1_gallery_match_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    # Mixed MSE + Mahalanobis scoring: z-score each independently, then blend.
    # alpha=1.0 → pure Mahal, alpha=0.0 → pure MSE, 0<alpha<1 → blend.
    if use_mahal and 0.0 < float(config.get("eval_mahal_alpha", 1.0)) < 1.0 and mahal_score_chunks:
        alpha = float(config.get("eval_mahal_alpha", 1.0))
        all_mahal = torch.cat(mahal_score_chunks, dim=0)
        # z-score each independently so they are on the same scale before blending
        def _zscore(s: torch.Tensor) -> torch.Tensor:
            return (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-6)
        all_scores = (1.0 - alpha) * _zscore(all_scores) + alpha * _zscore(all_mahal)

    f1_gallery_match_effective_alpha = 0.0
    if (
        bool(config.get("eval_f1_gallery_match_late_fusion", False))
        and all_f1_gallery_match_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        if all_scores.shape[1] > 1:
            a0_sorted = torch.sort(a0_scores, dim=1).values
            f1_sorted = torch.sort(all_f1_gallery_match_scores, dim=1).values
            a0_confidence = torch.clamp(
                (a0_sorted[:, 1] - a0_sorted[:, 0])
                / max(float(config.get("eval_f1_gallery_match_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
            f1_confidence = torch.clamp(
                (f1_sorted[:, 1] - f1_sorted[:, 0])
                / max(float(config.get("eval_f1_gallery_match_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            f1_confidence = torch.ones((all_scores.shape[0],), device=device)
        row_alpha = (
            float(config.get("eval_f1_gallery_match_late_alpha", 0.15))
            * f1_confidence
            * (1.0 - a0_confidence)
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        all_scores = (
            (1.0 - row_alpha) * a0_scores
            + row_alpha * all_f1_gallery_match_scores
        )
        f1_gallery_match_effective_alpha = float(row_alpha.mean().item())

    # Seen-class KNN bridge blending. Generic across ANY seen/unseen split —
    # bridge_alpha=0 disables it entirely (default), so this is a no-op unless
    # explicitly configured. As seen-class count shrinks (55/5 -> 30/30 ->
    # 12/48) the bridge signal gets noisier; tune/lower eval_bridge_alpha per
    # split rather than assuming one fixed value transfers across splits.
    bridge_alpha = float(config.get("eval_bridge_alpha", 0.0))
    bridge_fusion_stage = str(config.get("eval_bridge_fusion_stage", "early")).lower()
    if bridge_fusion_stage not in {"early", "late"}:
        raise ValueError(f"Unsupported eval_bridge_fusion_stage: {bridge_fusion_stage}")
    bridge_accuracy = -1.0
    bridge_effective_alpha = 0.0
    late_bridge_scores = None
    if bridge_ctx is not None and bridge_alpha > 0.0 and bridge_score_chunks:
        all_bridge = torch.cat(bridge_score_chunks, dim=0)
        bridge_mapped = torch.tensor(
            [label_to_idx[int(item)] for item in all_labels.detach().cpu()],
            device=device,
        )
        bridge_accuracy = float((torch.argmin(all_bridge, dim=1) == bridge_mapped).float().mean().item())
        if bridge_fusion_stage == "early":
            def _zscore_b(s: torch.Tensor) -> torch.Tensor:
                return (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-6)
            all_scores = (1.0 - bridge_alpha) * _zscore_b(all_scores) + bridge_alpha * _zscore_b(all_bridge)
            bridge_effective_alpha = bridge_alpha
        else:
            late_bridge_scores = all_bridge

    # Seen-analogy blending: borrows a discriminative axis from a UNIQUELY
    # matched seen "role-model" class per unseen class (bipartite matching on
    # text similarity), rather than pooling each candidate's own (possibly
    # heavily overlapping) neighbourhood. analogy_alpha=0 disables entirely.
    analogy_alpha = float(config.get("eval_analogy_alpha", 0.0))
    if analogy_ctx is not None and analogy_alpha > 0.0 and analogy_score_chunks:
        all_analogy = torch.cat(analogy_score_chunks, dim=0)
        def _zscore_g(s: torch.Tensor) -> torch.Tensor:
            return (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-6)
        all_scores = (1.0 - analogy_alpha) * _zscore_g(all_scores) + analogy_alpha * _zscore_g(all_analogy)

    # Transductive, label-free score calibration. This must be global over the
    # evaluation split; doing it per batch makes predictions depend on batch
    # composition and can erase rare / difficult unseen classes.
    if bool(config.get("eval_class_balance", False)):
        weight = float(config.get("eval_class_balance_weight", 1.0))
        all_scores = all_scores - all_scores.mean(dim=0, keepdim=True) * weight

    # A3 calibration: subtract per-class column mean to remove attractor bias.
    # This is the label-free transductive fix that prevents zero-acc collapse.
    if bool(config.get("eval_a3_calib", False)) and not eval_d6_evidence:
        all_scores = all_scores - all_scores.mean(dim=0, keepdim=True)

    if bool(config.get("eval_row_zscore", True)) and not eval_d6_evidence:
        all_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    # Transductive score refinement (eval_transductive: true).
    # Iteratively re-centres the score matrix using soft pseudo-label assignments,
    # so test samples that are neighbours in score-space pull each other toward
    # their shared class. No ground-truth labels are used.
    #
    # Algorithm (label-free, purely transductive):
    #   1. Convert current scores to soft weights via softmin (lower score = higher weight).
    #   2. Compute per-class weighted score centroid across ALL test samples.
    #   3. Subtract the centroid from each column (re-calibrate attractor bias).
    #   4. Re-apply row z-score.
    #   5. Repeat for n_iter iterations.
    if bool(config.get("eval_transductive", False)):
        n_iter = int(config.get("eval_transductive_iters", 5))
        temp = float(config.get("eval_transductive_temp", 1.0))
        for _ in range(n_iter):
            # Soft assignment weights: softmin over columns
            neg_scores = -all_scores / temp
            weights = torch.softmax(neg_scores, dim=1)           # (N, C) sums to 1 per row
            # Weighted column mean (pseudo-class centroid in score space)
            col_mean = (weights * all_scores).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-8)
            all_scores = all_scores - col_mean.unsqueeze(0)      # re-centre columns
            all_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(dim=1, keepdim=True).clamp_min(1e-6)

    # A0-safe late bridge fusion. A0 completes all global calibration first;
    # the prototype bridge may only re-rank A0's top-k candidates. Its maximum
    # weight is scaled by seen-only class-CV reliability and per-sample bridge
    # confidence, so unreliable mappings fall back toward pure A0.
    if late_bridge_scores is not None:
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        bridge_scores = late_bridge_scores
        if bool(config.get("eval_bridge_late_a3", True)):
            bridge_scores = bridge_scores - bridge_scores.mean(dim=0, keepdim=True)
        bridge_scores = (bridge_scores - bridge_scores.mean(dim=1, keepdim=True)) / bridge_scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        topk = max(1, min(int(config.get("eval_bridge_late_topk", 3)), all_scores.shape[1]))
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        bridge_candidates = torch.gather(bridge_scores, 1, candidate_indices)
        if topk > 1:
            sorted_bridge = torch.sort(bridge_candidates, dim=1).values
            bridge_margin = sorted_bridge[:, 1] - sorted_bridge[:, 0]
            confidence_scale = max(float(config.get("eval_bridge_confidence_scale", 0.5)), 1e-6)
            sample_confidence = torch.clamp(bridge_margin / confidence_scale, min=0.0, max=1.0)
        else:
            sample_confidence = torch.ones((all_scores.shape[0],), device=device)

        reliability = 1.0
        if bool(config.get("eval_bridge_reliability_gate", True)):
            reliability = float(bridge_ctx.get("cv_reliability", 0.0))
        row_alpha = torch.clamp(
            sample_confidence * bridge_alpha * reliability,
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        fused_candidates = (1.0 - row_alpha) * a0_candidates + row_alpha * bridge_candidates
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        bridge_effective_alpha = float(row_alpha.mean().item())

    mapped = torch.tensor(
        [label_to_idx[int(item)] for item in all_labels.detach().cpu()],
        device=device,
    )
    base_accuracy = (
        float((torch.argmin(all_scores, dim=1) == mapped).float().mean().item())
        if all_scores.shape[0]
        else 0.0
    )
    primitive_gate_data: dict[str, np.ndarray] | None = None
    primitive_utility_gate_data: dict[str, np.ndarray] | None = None
    if (
        bool(config.get("eval_collect_primitive_gate_data", False))
        and all_primitive_scores is not None
        and all_scores.shape[0]
    ):
        primitive_gate_data = {
            "a0_scores": all_scores.detach().float().cpu().numpy(),
            "primitive_scores": all_primitive_scores.detach().float().cpu().numpy(),
            "labels": all_labels.detach().cpu().numpy(),
            "candidate_labels": np.asarray(unseen_labels, dtype=np.int64),
        }
    primitive_effective_alpha = 0.0
    primitive_candidate_reliability = 1.0
    primitive_risk_gate_mean = 1.0
    primitive_utility_gate_enabled = 1.0
    if (
        bool(config.get("eval_primitive_late_fusion", False))
        and all_primitive_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        primitive_scores = all_primitive_scores
        topk = max(
            1,
            min(int(config.get("eval_primitive_late_topk", 3)), int(all_scores.shape[1])),
        )
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        primitive_candidates = torch.gather(primitive_scores, 1, candidate_indices)

        if all_scores.shape[1] > 1:
            sorted_a0 = torch.sort(a0_scores, dim=1).values
            sorted_primitive = torch.sort(primitive_scores, dim=1).values
            a0_margin = sorted_a0[:, 1] - sorted_a0[:, 0]
            primitive_margin = sorted_primitive[:, 1] - sorted_primitive[:, 0]
            a0_confidence = torch.clamp(
                a0_margin
                / max(float(config.get("eval_primitive_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
            primitive_confidence = torch.clamp(
                primitive_margin
                / max(float(config.get("eval_primitive_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            primitive_confidence = torch.ones((all_scores.shape[0],), device=device)

        primitive_top1 = torch.argmin(primitive_scores, dim=1, keepdim=True)
        primitive_in_a0_topk = candidate_indices.eq(primitive_top1).any(dim=1).to(
            dtype=primitive_confidence.dtype
        )
        row_alpha = (
            float(config.get("eval_primitive_late_alpha", 0.15))
            * primitive_confidence
            * (1.0 - a0_confidence)
            * primitive_in_a0_topk
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        nominal_candidates = (1.0 - row_alpha) * a0_candidates + row_alpha * primitive_candidates
        nominal_outside = a0_scores.max(dim=1, keepdim=True).values + 10.0
        nominal_scores = nominal_outside.expand_as(a0_scores).clone()
        nominal_scores.scatter_(1, candidate_indices, nominal_candidates)
        utility_features = primitive_utility_gate_features(
            a0_scores,
            primitive_scores,
            a0_candidates,
            nominal_candidates,
        )
        if bool(config.get("eval_collect_primitive_utility_gate_data", False)):
            primitive_utility_gate_data = {
                "a0_scores": a0_scores.detach().float().cpu().numpy(),
                "nominal_fused_scores": nominal_scores.detach().float().cpu().numpy(),
                "utility_features": utility_features.detach().float().cpu().numpy(),
                "labels": all_labels.detach().cpu().numpy(),
                "candidate_labels": np.asarray(unseen_labels, dtype=np.int64),
            }
        utility_gate_path = config.get("eval_primitive_utility_gate_path")
        if utility_gate_path:
            utility_gate = load_primitive_utility_gate(utility_gate_path, device)
            utility_logits = ((utility_features.float() - utility_gate["mean"]) / utility_gate["scale"]).matmul(
                utility_gate["coef"]
            ) + utility_gate["intercept"]
            utility_probability = torch.sigmoid(utility_logits)
            threshold = min(
                max(float(config.get("eval_primitive_utility_gate_threshold", 0.5)), 0.0),
                1.0,
            )
            utility_enabled = utility_probability.ge(threshold).to(dtype=row_alpha.dtype)
            row_alpha = row_alpha * utility_enabled.unsqueeze(1)
            primitive_utility_gate_enabled = float(utility_enabled.mean().item())
        risk_gate_path = config.get("eval_primitive_risk_gate_path")
        if risk_gate_path:
            gate = load_primitive_risk_gate(risk_gate_path, device)
            gate_features = primitive_risk_gate_features(a0_scores, primitive_scores).float()
            gate_logits = ((gate_features - gate["mean"]) / gate["scale"]).matmul(
                gate["coef"]
            ) + gate["intercept"]
            gate_probability = torch.sigmoid(gate_logits).clamp(0.0, 1.0)
            row_alpha = row_alpha * gate_probability.unsqueeze(1)
            primitive_risk_gate_mean = float(gate_probability.mean().item())
        candidate_alpha = row_alpha
        if bool(config.get("eval_primitive_seen_support_gate", False)):
            candidate_reliability = primitive_ctx["class_reliability"].to(
                device=device,
                dtype=a0_candidates.dtype,
            )[torch.as_tensor(unseen_labels, device=device, dtype=torch.long)]
            candidate_reliability = torch.gather(
                candidate_reliability.unsqueeze(0).expand_as(a0_scores),
                1,
                candidate_indices,
            )
            reliability_floor = min(
                max(float(config.get("eval_primitive_reliability_floor", 0.0)), 0.0),
                1.0,
            )
            candidate_reliability = candidate_reliability.clamp_min(reliability_floor)
            candidate_alpha = row_alpha * candidate_reliability
            primitive_candidate_reliability = float(candidate_reliability.mean().item())
        fused_candidates = (
            (1.0 - candidate_alpha) * a0_candidates
            + candidate_alpha * primitive_candidates
        )
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        primitive_effective_alpha = float(candidate_alpha.mean().item())

    # D5 clean-token scores are a separate view of the frozen skeleton
    # feature.  They only re-rank the already calibrated C10/C21 candidates
    # when both views are uncertain; this prevents the clean branch from
    # replacing the conditional DiT energy.
    clean_tide_effective_alpha = 0.0
    if (
        bool(config.get("eval_clean_tide_late_fusion", False))
        and all_clean_tide_primitive_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        clean_scores = all_clean_tide_primitive_scores
        topk = max(
            1,
            min(int(config.get("eval_clean_tide_late_topk", 3)), int(all_scores.shape[1])),
        )
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        clean_candidates = torch.gather(clean_scores, 1, candidate_indices)
        if all_scores.shape[1] > 1:
            sorted_a0 = torch.sort(a0_scores, dim=1).values
            sorted_clean = torch.sort(clean_scores, dim=1).values
            a0_confidence = torch.clamp(
                (sorted_a0[:, 1] - sorted_a0[:, 0])
                / max(float(config.get("eval_clean_tide_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
            clean_confidence = torch.clamp(
                (sorted_clean[:, 1] - sorted_clean[:, 0])
                / max(float(config.get("eval_clean_tide_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            clean_confidence = torch.ones((all_scores.shape[0],), device=device)
        clean_top1 = torch.argmin(clean_scores, dim=1, keepdim=True)
        clean_in_a0_topk = candidate_indices.eq(clean_top1).any(dim=1).to(clean_confidence.dtype)
        row_alpha = (
            float(config.get("eval_clean_tide_late_alpha", 0.15))
            * clean_confidence
            * (1.0 - a0_confidence)
            * clean_in_a0_topk
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        fused_candidates = (1.0 - row_alpha) * a0_candidates + row_alpha * clean_candidates
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        clean_tide_effective_alpha = float(row_alpha.mean().item())

    feature_distribution_effective_alpha = 0.0
    if (
        bool(config.get("eval_primitive_feature_distribution_late_fusion", False))
        and all_feature_distribution_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        distribution_scores = all_feature_distribution_scores
        topk = max(
            1,
            min(
                int(config.get("eval_primitive_feature_distribution_late_topk", 3)),
                int(all_scores.shape[1]),
            ),
        )
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        distribution_candidates = torch.gather(distribution_scores, 1, candidate_indices)
        if all_scores.shape[1] > 1:
            sorted_a0 = torch.sort(a0_scores, dim=1).values
            sorted_distribution = torch.sort(distribution_scores, dim=1).values
            a0_margin = sorted_a0[:, 1] - sorted_a0[:, 0]
            distribution_margin = sorted_distribution[:, 1] - sorted_distribution[:, 0]
            a0_confidence = torch.clamp(
                a0_margin
                / max(float(config.get("eval_primitive_feature_distribution_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
            distribution_confidence = torch.clamp(
                distribution_margin
                / max(float(config.get("eval_primitive_feature_distribution_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            distribution_confidence = torch.ones((all_scores.shape[0],), device=device)
        distribution_top1 = torch.argmin(distribution_scores, dim=1, keepdim=True)
        distribution_in_a0_topk = candidate_indices.eq(distribution_top1).any(dim=1).to(
            dtype=distribution_confidence.dtype
        )
        row_alpha = (
            float(config.get("eval_primitive_feature_distribution_late_alpha", 0.10))
            * distribution_confidence
            * (1.0 - a0_confidence)
            * distribution_in_a0_topk
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        fused_candidates = (
            (1.0 - row_alpha) * a0_candidates
            + row_alpha * distribution_candidates
        )
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        feature_distribution_effective_alpha = float(row_alpha.mean().item())

    episodic_proto_effective_alpha = 0.0
    if (
        bool(config.get("eval_episodic_proto_late_fusion", False))
        and all_episodic_proto_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        proto_scores = all_episodic_proto_scores
        topk = max(1, min(int(config.get("eval_episodic_proto_late_topk", 3)), int(all_scores.shape[1])))
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        proto_candidates = torch.gather(proto_scores, 1, candidate_indices)
        if all_scores.shape[1] > 1:
            a0_margin = torch.sort(a0_scores, dim=1).values[:, 1] - torch.sort(a0_scores, dim=1).values[:, 0]
            proto_margin = torch.sort(proto_scores, dim=1).values[:, 1] - torch.sort(proto_scores, dim=1).values[:, 0]
            a0_confidence = torch.clamp(
                a0_margin / max(float(config.get("eval_episodic_proto_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0, max=1.0,
            )
            proto_confidence = torch.clamp(
                proto_margin / max(float(config.get("eval_episodic_proto_confidence_scale", 0.5)), 1e-6),
                min=0.0, max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            proto_confidence = torch.ones((all_scores.shape[0],), device=device)
        proto_top1 = torch.argmin(proto_scores, dim=1, keepdim=True)
        proto_in_a0_topk = candidate_indices.eq(proto_top1).any(dim=1).to(dtype=proto_confidence.dtype)
        row_alpha = (
            float(config.get("eval_episodic_proto_late_alpha", 0.15))
            * episodic_proto_reliability
            * proto_confidence
            * (1.0 - a0_confidence)
            * proto_in_a0_topk
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        fused_candidates = (1.0 - row_alpha) * a0_candidates + row_alpha * proto_candidates
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        episodic_proto_effective_alpha = float(row_alpha.mean().item())

    proj_effective_alpha = 0.0
    if (
        bool(config.get("eval_proj_late_fusion", False))
        and all_proj_scores is not None
        and all_scores.shape[0]
    ):
        a0_scores = (all_scores - all_scores.mean(dim=1, keepdim=True)) / all_scores.std(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        proj_scores = all_proj_scores
        topk = max(
            1,
            min(int(config.get("eval_proj_late_topk", 3)), int(all_scores.shape[1])),
        )
        candidate_indices = torch.topk(a0_scores, k=topk, dim=1, largest=False).indices
        a0_candidates = torch.gather(a0_scores, 1, candidate_indices)
        proj_candidates = torch.gather(proj_scores, 1, candidate_indices)

        if all_scores.shape[1] > 1:
            sorted_a0 = torch.sort(a0_scores, dim=1).values
            sorted_proj = torch.sort(proj_scores, dim=1).values
            a0_margin = sorted_a0[:, 1] - sorted_a0[:, 0]
            proj_margin = sorted_proj[:, 1] - sorted_proj[:, 0]
            a0_confidence = torch.clamp(
                a0_margin / max(float(config.get("eval_proj_a0_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
            proj_confidence = torch.clamp(
                proj_margin / max(float(config.get("eval_proj_confidence_scale", 0.5)), 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            a0_confidence = torch.zeros((all_scores.shape[0],), device=device)
            proj_confidence = torch.ones((all_scores.shape[0],), device=device)

        proj_top1 = torch.argmin(proj_scores, dim=1, keepdim=True)
        proj_in_a0_topk = candidate_indices.eq(proj_top1).any(dim=1).to(dtype=proj_confidence.dtype)
        row_alpha = (
            float(config.get("eval_proj_late_alpha", 0.15))
            * proj_confidence
            * (1.0 - a0_confidence)
            * proj_in_a0_topk
        ).clamp(min=0.0, max=1.0).unsqueeze(1)
        fused_candidates = (1.0 - row_alpha) * a0_candidates + row_alpha * proj_candidates
        outside_value = a0_scores.max(dim=1, keepdim=True).values + 10.0
        fused_scores = outside_value.expand_as(a0_scores).clone()
        fused_scores.scatter_(1, candidate_indices, fused_candidates)
        all_scores = fused_scores
        proj_effective_alpha = float(row_alpha.mean().item())

    feature_graph_effective_alpha = 0.0
    if bool(config.get("eval_feature_graph", False)) and feature_chunks:
        all_features = torch.cat(feature_chunks, dim=0)
        all_scores, feature_graph_effective_alpha = refine_scores_on_frozen_feature_graph(
            all_scores,
            all_features,
            config,
        )

    ot_enabled = bool(config.get("eval_transductive_ot", False))
    semi_uot_enabled = bool(config.get("eval_semi_uot", False))
    if ot_enabled and semi_uot_enabled:
        raise ValueError("Choose either eval_transductive_ot or eval_semi_uot, not both")
    ot_epsilon = float(config.get("eval_ot_epsilon", 0.5))
    ot_iterations = int(config.get("eval_ot_iterations", 100))
    ot_marginal_error = 0.0
    semi_uot_epsilon = float(config.get("eval_semi_uot_epsilon", 1.0))
    semi_uot_tau = float(config.get("eval_semi_uot_target_tau", 0.10))
    semi_uot_iterations = int(config.get("eval_semi_uot_iterations", 100))
    semi_uot_source_error = 0.0
    semi_uot_target_kl = 0.0
    if semi_uot_enabled:
        semi_uot_plan = semi_unbalanced_sinkhorn_transport(
            all_scores,
            epsilon=semi_uot_epsilon,
            target_tau=semi_uot_tau,
            iterations=semi_uot_iterations,
        )
        source_mass = 1.0 / float(semi_uot_plan.shape[0])
        target_mass = semi_uot_plan.sum(dim=0)
        uniform_target = 1.0 / float(semi_uot_plan.shape[1])
        semi_uot_source_error = float(
            (semi_uot_plan.sum(dim=1) - source_mass).abs().max().detach().cpu()
        )
        semi_uot_target_kl = float(
            (
                target_mass
                * (target_mass.clamp_min(1e-12) / uniform_target).log()
            ).sum().detach().cpu()
        )
        pred_cols = torch.argmax(semi_uot_plan, dim=1)
    elif ot_enabled:
        ot_plan = balanced_sinkhorn_transport(
            all_scores,
            epsilon=ot_epsilon,
            iterations=ot_iterations,
        )
        target_mass = 1.0 / float(ot_plan.shape[1])
        ot_marginal_error = float(
            (ot_plan.sum(dim=0) - target_mass).abs().max().detach().cpu()
        )
        pred_cols = torch.argmax(ot_plan, dim=1)
    else:
        pred_cols = torch.argmin(all_scores, dim=1)
    confusion = torch.zeros((len(unseen_labels), len(unseen_labels)), device=device, dtype=torch.long)
    for true_idx, pred_idx in zip(mapped.view(-1), pred_cols.view(-1)):
        confusion[true_idx.long(), pred_idx.long()] += 1
    total = int(all_labels.numel())
    correct = int((pred_cols == mapped).sum().item()) if total else 0

    # Projection-space secondary accuracy (only when eval_also_proj_score).
    proj_acc: float = -1.0
    if all_proj_scores is not None:
        proj_pred = torch.argmin(all_proj_scores, dim=1)
        proj_acc = float((proj_pred == mapped).sum().item()) / total if total else 0.0
    f1_gallery_match_acc: float = -1.0
    if all_f1_gallery_match_scores is not None:
        f1_gallery_match_pred = torch.argmin(all_f1_gallery_match_scores, dim=1)
        f1_gallery_match_acc = (
            float((f1_gallery_match_pred == mapped).sum().item()) / total if total else 0.0
        )
    primitive_acc: float = -1.0
    if all_primitive_scores is not None:
        primitive_pred = torch.argmin(all_primitive_scores, dim=1)
        primitive_acc = (
            float((primitive_pred == mapped).sum().item()) / total if total else 0.0
        )
    clean_tide_primitive_acc: float = -1.0
    if all_clean_tide_primitive_scores is not None:
        clean_tide_pred = torch.argmin(all_clean_tide_primitive_scores, dim=1)
        clean_tide_primitive_acc = (
            float((clean_tide_pred == mapped).sum().item()) / total if total else 0.0
        )
    feature_distribution_acc: float = -1.0
    if all_feature_distribution_scores is not None:
        feature_distribution_pred = torch.argmin(all_feature_distribution_scores, dim=1)
        feature_distribution_acc = (
            float((feature_distribution_pred == mapped).sum().item()) / total if total else 0.0
        )
    episodic_proto_acc: float = -1.0
    if all_episodic_proto_scores is not None:
        episodic_proto_pred = torch.argmin(all_episodic_proto_scores, dim=1)
        episodic_proto_acc = (
            float((episodic_proto_pred == mapped).sum().item()) / total if total else 0.0
        )

    elapsed = time.time() - start_time
    return {
        "accuracy": correct / total if total else 0.0,
        "base_accuracy": base_accuracy,
        "proj_accuracy": proj_acc,
        "proj_effective_alpha": proj_effective_alpha,
        "f1_gallery_match_accuracy": f1_gallery_match_acc,
        "f1_gallery_match_effective_alpha": f1_gallery_match_effective_alpha,
        "primitive_accuracy": primitive_acc,
        "primitive_effective_alpha": primitive_effective_alpha,
        "primitive_candidate_reliability": primitive_candidate_reliability,
        "primitive_risk_gate_mean": primitive_risk_gate_mean,
        "primitive_gate_data": primitive_gate_data,
        "primitive_utility_gate_enabled": primitive_utility_gate_enabled,
        "primitive_utility_gate_data": primitive_utility_gate_data,
        "clean_tide_primitive_accuracy": clean_tide_primitive_acc,
        "clean_tide_effective_alpha": clean_tide_effective_alpha,
        "feature_distribution_accuracy": feature_distribution_acc,
        "feature_distribution_effective_alpha": feature_distribution_effective_alpha,
        "episodic_proto_accuracy": episodic_proto_acc,
        "episodic_proto_seen_accuracy": episodic_proto_seen_accuracy,
        "episodic_proto_reliability": episodic_proto_reliability,
        "episodic_proto_effective_alpha": episodic_proto_effective_alpha,
        "feature_graph_enabled": bool(config.get("eval_feature_graph", False)),
        "feature_graph_effective_alpha": feature_graph_effective_alpha,
        "feature_graph_k": int(config.get("eval_feature_graph_k", 0)) if bool(config.get("eval_feature_graph", False)) else 0,
        "feature_graph_iters": int(config.get("eval_feature_graph_iters", 0)) if bool(config.get("eval_feature_graph", False)) else 0,
        "d6_evidence_enabled": eval_d6_evidence,
        "d6_timesteps": [int(value) for value in t_values] if eval_d6_evidence else [],
        "d6_weights": d6_weights.detach().cpu().tolist() if d6_weights is not None else [],
        "d6_weight_source": d6_metadata.get("source", None) if d6_metadata is not None else None,
        "adaptive_timestep_enabled": eval_adaptive_timestep,
        "adaptive_timestep_timesteps": adaptive_metadata.get("timesteps", []) if adaptive_metadata else [],
        "adaptive_timestep_mean_weights": adaptive_metadata.get("mean_weights", []) if adaptive_metadata else [],
        "adaptive_timestep_mean_evidence": adaptive_metadata.get("mean_evidence", []) if adaptive_metadata else [],
        "adaptive_timestep_mean_margin": adaptive_metadata.get("mean_margin", []) if adaptive_metadata else [],
        "adaptive_timestep_mean_variance_penalty": adaptive_metadata.get("mean_variance_penalty", []) if adaptive_metadata else [],
        "adaptive_timestep_weight_entropy": adaptive_metadata.get("weight_entropy", 0.0) if adaptive_metadata else 0.0,
        "hetero_multiview_enabled": eval_hetero_multiview,
        "hetero_timesteps": t_values if eval_hetero_multiview else [],
        "hetero_mean_weights": hetero_metadata.get("mean_weights", []) if hetero_metadata else [],
        "hetero_mean_relative_variance": hetero_metadata.get("mean_relative_variance", []) if hetero_metadata else [],
        "hetero_correlation": hetero_metadata.get("correlation", []) if hetero_metadata else [],
        "hetero_negative_weight_fraction": float(
            hetero_metadata.get("negative_weight_fraction", 0.0) if hetero_metadata else 0.0
        ),
        "ot_enabled": ot_enabled,
        "ot_epsilon": ot_epsilon if ot_enabled else 0.0,
        "ot_iterations": ot_iterations if ot_enabled else 0,
        "ot_marginal_error": ot_marginal_error,
        "semi_uot_enabled": semi_uot_enabled,
        "semi_uot_epsilon": semi_uot_epsilon if semi_uot_enabled else 0.0,
        "semi_uot_target_tau": semi_uot_tau if semi_uot_enabled else 0.0,
        "semi_uot_iterations": semi_uot_iterations if semi_uot_enabled else 0,
        "semi_uot_source_marginal_error": semi_uot_source_error,
        "semi_uot_target_uniform_kl": semi_uot_target_kl,
        "cfg_scale": float(config.get("eval_cfg_scale", 1.0)),
        "use_mahal": use_mahal,
        "bridge_alpha": bridge_alpha,
        "bridge_effective_alpha": bridge_effective_alpha,
        "bridge_accuracy": bridge_accuracy,
        "bridge_mode": bridge_ctx.get("mode") if bridge_ctx is not None else None,
        "bridge_fusion_stage": bridge_fusion_stage if bridge_ctx is not None else None,
        "bridge_reliability": float(bridge_ctx.get("cv_reliability", 0.0)) if bridge_ctx is not None else 0.0,
        "num_samples": total,
        "elapsed_sec": elapsed,
        "samples_per_sec": total / elapsed if elapsed > 0 else 0.0,
        "unseen_labels": [int(item) for item in unseen_labels],
        "eval_timesteps": [int(item) for item in t_values],
        "eval_reverse_x0_enabled": reverse_x0_timesteps is not None,
        "eval_reverse_x0_timesteps": reverse_x0_timesteps or [],
        "eval_num_noise": int(noise_count),
        "eval_noise_seed": eval_seed,
        "eval_noise_mode": eval_noise_mode,
        "eval_distance_space": "prediction" if use_native_eval_distance else "x0",
        "confusion_matrix": confusion.detach().cpu().numpy(),
        # The evaluation loader is non-shuffled, so these arrays preserve the
        # dataset order and allow post-hoc, sample-level model comparisons.
        "true_labels": all_labels.detach().cpu().numpy(),
        "predicted_labels": unseen_labels[pred_cols.detach().cpu().numpy()],
    }


def save_eval_artifacts(work_dir: Path, metrics: dict, tag: str) -> None:
    labels = metrics["unseen_labels"]
    confusion = metrics["confusion_matrix"]
    with (work_dir / f"confusion_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label\\pred_label"] + labels)
        for label, row in zip(labels, confusion):
            writer.writerow([label] + [int(value) for value in row])

    with (work_dir / f"per_class_acc_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "correct", "total", "accuracy"])
        for idx, label in enumerate(labels):
            total = int(np.sum(confusion[idx]))
            correct = int(confusion[idx, idx])
            writer.writerow([label, correct, total, f"{correct / total if total else 0.0:.6f}"])

    true_labels = metrics.get("true_labels")
    predicted_labels = metrics.get("predicted_labels")
    if true_labels is not None and predicted_labels is not None:
        with (work_dir / f"predictions_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "true_label", "predicted_label", "correct"])
            for sample_index, (true_label, predicted_label) in enumerate(
                zip(true_labels, predicted_labels)
            ):
                writer.writerow(
                    [
                        sample_index,
                        int(true_label),
                        int(predicted_label),
                        int(true_label == predicted_label),
                    ]
                )


def train(config: dict, args: argparse.Namespace) -> None:
    init_seed(int(config.get("seed", 2025)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    work_dir = repo_path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_loader = make_loader(config, "train", shuffle=True, max_samples=int(args.max_train_samples))
    test_loader = make_loader(config, "test", shuffle=False, max_samples=int(args.max_test_samples))
    if config.get("unseen_labels_override", None) is not None:
        unseen_labels = np.asarray(config["unseen_labels_override"], dtype=np.int64).reshape(-1)
    else:
        unseen_labels = np.load(repo_path(config["unseen_label_path"])).astype(np.int64)

    # Derive the negative-label pool from samples that are actually available
    # for training. This is essential for class-disjoint pseudo-unseen
    # validation, where all_classes - validation_classes may still contain
    # official unseen classes that have no training samples.
    train_class_labels = np.unique(loader_label_array(train_loader))
    test_class_labels = np.unique(loader_label_array(test_loader))
    if bool(config.get("strict_zsl_split", True)):
        leaked = np.intersect1d(train_class_labels, unseen_labels)
        unexpected_test = np.setdiff1d(test_class_labels, unseen_labels)
        if leaked.size:
            raise ValueError(
                "ZSL split leakage: unseen labels occur in the training data: "
                f"{leaked.tolist()}"
            )
        if unexpected_test.size:
            raise ValueError(
                "ZSL test split contains labels outside unseen_label_path: "
                f"{unexpected_test.tolist()}"
            )

    model_path = config.get("pretrained_model_name_or_path", "sd2-community/stable-diffusion-2-1")
    text_embed = build_text_embed(
        model_path,
        device,
        dtype,
        local_files_only=bool(config.get("local_files_only", True)),
        config=config,
    )
    before_text_stats = text_similarity_stats(text_embed, unseen_labels)
    text_embed, text_transform_info = transform_text_embed(text_embed, config)
    after_text_stats = text_similarity_stats(text_embed, unseen_labels)
    num_classes = int(text_embed.shape[0])
    configured_cond_size = int(config.get("cond_size", text_embed.shape[-1]))
    if configured_cond_size != int(text_embed.shape[-1]):
        raise ValueError(
            f"cond_size={configured_cond_size} does not match selected text dimension "
            f"{int(text_embed.shape[-1])} for clip_text_source={config.get('clip_text_source', 'concat')}"
        )
    if np.any(train_class_labels < 0) or np.any(train_class_labels >= num_classes):
        raise ValueError(
            f"Training labels must be in [0, {num_classes - 1}], got "
            f"{train_class_labels.tolist()}"
        )
    seen_labels = train_class_labels
    scheduler = build_scheduler(config, local_files_only=bool(config.get("local_files_only", True)))
    model_type = str(config.get("model_type", "original")).lower()
    model_kwargs = {
        "in_channels": int(config.get("in_channels", 256)),
        "cond_size": int(config.get("cond_size", text_embed.shape[-1])),
        "hidden_size": int(config["hidden_size"]),
        "depth": int(config["depth"]),
        "num_heads": int(config["num_heads"]),
    }
    if model_type in {"fine_grained", "finegrained", "fg"}:
        ModelClass = DiTFineGrained
        model_kwargs["feature_tokens"] = int(config.get("feature_tokens", 8))
        model_kwargs["text_fusion"] = str(config.get("text_fusion", "channel"))
        model_kwargs["use_text_mask"] = bool(config.get("mask_text_padding", False))
        model_kwargs["text_detail_layers"] = config.get("text_detail_layers", None)
        model_kwargs["agreement_topk"] = int(config.get("agreement_topk", 8))
        model_kwargs["agreement_threshold"] = float(config.get("agreement_threshold", 0.6))
        model_kwargs["agreement_temperature"] = float(config.get("agreement_temperature", 0.15))
        model_kwargs["agreement_detail_gate_init"] = float(
            config.get("agreement_detail_gate_init", 0.0)
        )
        model_kwargs["agreement_zero_out_proj"] = bool(
            config.get("agreement_zero_out_proj", False)
        )
        model_kwargs["enable_self_attention"] = bool(
            config.get("enable_self_attention", True)
        )
        model_kwargs["enable_cross_attention"] = bool(
            config.get("enable_cross_attention", True)
        )
        model_kwargs["enable_global_text"] = bool(
            config.get("enable_global_text", True)
        )
        model_kwargs["text_injection_layers"] = config.get("text_injection_layers")
        model_kwargs["text_injection_final"] = bool(
            config.get("text_injection_final", True)
        )
    elif model_type == "text_gate":
        ModelClass = DiTTextGate
    else:
        ModelClass = DiT
    model = ModelClass(
        **model_kwargs,
    ).to(device=device, dtype=dtype)

    loss_mode = str(config.get("loss_mode", "c2u_feat")).lower()
    semantic_distill_modes = {"tdsm_x0_semantic_distill", "tdsm_x0_topology"}
    prototype_direction_modes = {"tdsm_x0_proto_direction", "tdsm_x0_prototype_direction"}
    adaptive_margin_modes = {"tdsm_x0_adaptive_margin", "tdsm_x0_geometry_margin"}
    noise_consistent_modes = {"tdsm_x0_noise_consistent", "tdsm_x0_noise_lcb"}
    agreement_triplet_modes = {"tdsm_x0_agreement_weighted_triplet"}
    timestep_gap_invariance_modes = {
        "tdsm_x0_action_primitive_timestep_gap",
        "tdsm_x0_action_primitive_timestep_reliable_gap",
    }
    loss_aware_timestep_modes = {"tdsm_x0_action_primitive_loss_aware_sampler"}
    teacher_timestep_modes = {"tdsm_x0_teacher_timestep", "tdsm_x0_adaptive_timestep"}
    adaptive_evidence_timestep_modes = {
        "tdsm_x0_action_primitive_adaptive_evidence",
        "tdsm_x0_action_primitive_timestep_policy_distill",
    }
    episodic_gallery_modes = {"tdsm_x0_action_primitive_episodic_gallery"}
    ambiguity_contrast_modes = {"tdsm_x0_ambiguity_contrast", "tdsm_x0_false_negative_aware"}
    natural_ambiguity_modes = {"tdsm_x0_natural_ambiguity_margin"}
    dual_random_modes = {"tdsm_x0_dual_random_triplet", "tdsm_x0_multi_random_triplet"}
    multipos_align_modes = {"tdsm_x0_multipos_align", "tdsm_x0_bidirectional_multipos"}
    neutral_align_modes = {
        "tdsm_x0_neutral_debiased_align",
        "tdsm_x0_neutral_semantic_contrast",
    }
    primitive_multilabel_modes = {
        "tdsm_x0_action_primitive_multilabel",
        "tdsm_x0_latent_primitive_multilabel",
        "tdsm_x0_action_primitive_proto_contrast",
        "tdsm_x0_action_primitive_tmr_filtered",
        "tdsm_x0_action_primitive_feature_distribution",
        "tdsm_x0_action_primitive_episodic_proto",
        "tdsm_x0_action_primitive_episodic_gallery",
        "tdsm_x0_action_primitive_counterfactual_text",
        "tdsm_x0_action_primitive_energy_calibration",
        "tdsm_x0_action_primitive_ordinal_geometry",
        "tdsm_x0_action_primitive_ordinal_geometry_reliable",
        "tdsm_x0_action_primitive_soft_ordinal_geometry",
        "tdsm_x0_action_primitive_semi_uot",
        "tdsm_x0_action_primitive_nnpu",
        "tdsm_x0_action_primitive_band_elbo_snr",
        "tdsm_x0_action_primitive_timestep_gap",
        "tdsm_x0_action_primitive_timestep_reliable_gap",
        "tdsm_x0_action_primitive_adaptive_evidence",
        "tdsm_x0_action_primitive_timestep_policy_distill",
        "tdsm_x0_action_primitive_loss_aware_sampler",
        "tdsm_x0_action_primitive_multitime_energy_distill",
        "tdsm_x0_action_primitive_clean_tide",
        "tdsm_x0_action_primitive_clean_tide_multilayer",
        "tdsm_x0_action_primitive_frozen_gallery_match",
    }
    use_semantic_distill = loss_mode in semantic_distill_modes
    use_prototype_direction = loss_mode in prototype_direction_modes
    use_adaptive_margin = loss_mode in adaptive_margin_modes
    use_noise_consistent = loss_mode in noise_consistent_modes
    use_agreement_weighted_triplet = loss_mode in agreement_triplet_modes
    use_timestep_gap_invariance = loss_mode in timestep_gap_invariance_modes
    use_reliable_timestep_gap = loss_mode == "tdsm_x0_action_primitive_timestep_reliable_gap"
    use_loss_aware_timestep = loss_mode in loss_aware_timestep_modes
    use_teacher_timestep = loss_mode in teacher_timestep_modes
    use_adaptive_evidence_timestep = loss_mode in adaptive_evidence_timestep_modes
    use_timestep_policy_distill = loss_mode == "tdsm_x0_action_primitive_timestep_policy_distill"
    use_episodic_gallery = loss_mode in episodic_gallery_modes
    use_multitime_energy_distill = (
        loss_mode == "tdsm_x0_action_primitive_multitime_energy_distill"
    )
    use_ambiguity_contrast = loss_mode in ambiguity_contrast_modes
    use_natural_ambiguity = loss_mode in natural_ambiguity_modes
    use_dual_random = loss_mode in dual_random_modes
    use_multipos_align = loss_mode in multipos_align_modes
    use_neutral_align = loss_mode in neutral_align_modes
    use_primitive_multilabel = loss_mode in primitive_multilabel_modes
    use_clean_tide = loss_mode == "tdsm_x0_action_primitive_clean_tide"
    use_clean_tide_multilayer = loss_mode == "tdsm_x0_action_primitive_clean_tide_multilayer"
    use_primitive_proto_contrast = loss_mode == "tdsm_x0_action_primitive_proto_contrast"
    use_primitive_tmr_filtered = loss_mode == "tdsm_x0_action_primitive_tmr_filtered"
    use_primitive_feature_distribution = (
        loss_mode == "tdsm_x0_action_primitive_feature_distribution"
    )
    use_episodic_primitive_proto = (
        loss_mode == "tdsm_x0_action_primitive_episodic_proto"
    )
    use_counterfactual_text = (
        loss_mode == "tdsm_x0_action_primitive_counterfactual_text"
    )
    use_energy_calibration = (
        loss_mode == "tdsm_x0_action_primitive_energy_calibration"
    )
    use_ordinal_geometry = loss_mode in {
        "tdsm_x0_action_primitive_ordinal_geometry",
        "tdsm_x0_action_primitive_ordinal_geometry_reliable",
        "tdsm_x0_action_primitive_soft_ordinal_geometry",
    }
    use_reliable_ordinal_geometry = (
        loss_mode == "tdsm_x0_action_primitive_ordinal_geometry_reliable"
    )
    use_soft_ordinal_geometry = (
        loss_mode == "tdsm_x0_action_primitive_soft_ordinal_geometry"
    )
    use_primitive_semi_uot = loss_mode == "tdsm_x0_action_primitive_semi_uot"
    use_primitive_nnpu = loss_mode == "tdsm_x0_action_primitive_nnpu"
    use_band_elbo_snr = loss_mode == "tdsm_x0_action_primitive_band_elbo_snr"
    use_frozen_gallery_match = (
        loss_mode == "tdsm_x0_action_primitive_frozen_gallery_match"
    )
    primitive_gradient_mode = str(config.get("primitive_gradient_mode", "standard")).lower()
    use_primitive_pcgrad = primitive_gradient_mode == "pcgrad"
    primitive_pcgrad_max_ratio = float(config.get("primitive_pcgrad_max_ratio", 0.05))
    if primitive_gradient_mode not in {"standard", "pcgrad"}:
        raise ValueError(f"Unsupported primitive_gradient_mode: {primitive_gradient_mode}")
    if use_primitive_pcgrad and not use_primitive_multilabel:
        raise ValueError("primitive_gradient_mode=pcgrad requires a primitive multi-label loss mode")
    if use_primitive_pcgrad and bool(config.get("primitive_detach_backbone", True)):
        raise ValueError("primitive_gradient_mode=pcgrad requires primitive_detach_backbone=false")
    if use_primitive_pcgrad and primitive_pcgrad_max_ratio <= 0.0:
        raise ValueError("primitive_pcgrad_max_ratio must be positive")
    if use_primitive_nnpu:
        prior_floor = float(config.get("primitive_pu_prior_floor", 0.01))
        prior_ceiling = float(config.get("primitive_pu_prior_ceiling", 0.99))
        if not 0.0 <= prior_floor <= prior_ceiling <= 1.0:
            raise ValueError("nnPU primitive priors must satisfy 0 <= floor <= ceiling <= 1")
    if use_frozen_gallery_match:
        pooled_text_size = int(pooled_text_embed(text_embed).shape[-1])
        if pooled_text_size % 2 != 0:
            raise ValueError("F1 requires concatenated CSV/LLM pooled text features")
    elbo_sampling_probabilities = None
    elbo_objective_weights = None
    elbo_schedule_metadata = None
    if use_band_elbo_snr:
        elbo_sampling_probabilities, elbo_objective_weights, elbo_schedule_metadata = (
            build_band_tempered_elbo_schedule(
                scheduler=scheduler,
                start_timestep=int(config.get("train_timestep_min", 1)),
                end_timestep=int(config.get("train_timestep_max", int(config["num_steps"]) - 1)),
                num_bands=int(config.get("elbo_snr_num_bands", 3)),
                power=float(config.get("elbo_snr_power", 0.25)),
                device=device,
            )
        )
    noise_consistent_views = int(config.get("noise_consistent_views", 2))
    noise_consistent_beta = float(config.get("noise_consistent_beta", 0.5))
    if use_noise_consistent and noise_consistent_views < 2:
        raise ValueError("noise_consistent_views must be at least 2")
    if use_noise_consistent and noise_consistent_beta < 0.0:
        raise ValueError("noise_consistent_beta must be non-negative")
    loss_aware_ema_decay = float(config.get("loss_aware_ema_decay", 0.99))
    loss_aware_uniform_mix = float(config.get("loss_aware_uniform_mix", 0.10))
    loss_aware_start_step = max(0, int(config.get("loss_aware_start_step", 500)))
    if use_loss_aware_timestep and not 0.0 <= loss_aware_ema_decay < 1.0:
        raise ValueError("loss_aware_ema_decay must be in [0, 1)")
    if use_loss_aware_timestep and not 0.0 <= loss_aware_uniform_mix <= 1.0:
        raise ValueError("loss_aware_uniform_mix must be in [0, 1]")
    timestep_gap_anchor = int(config.get("timestep_gap_anchor", 25))
    default_gap_partners = [30, 40] if use_reliable_timestep_gap else [10, 40]
    timestep_gap_partners = [
        int(value) for value in config.get("timestep_gap_partner_timesteps", default_gap_partners)
    ]
    default_gap_weights = [0.5, 0.5] if use_reliable_timestep_gap else [1.0] * len(timestep_gap_partners)
    timestep_gap_partner_weights = [
        float(value)
        for value in config.get("timestep_gap_partner_weights", default_gap_weights)
    ]
    default_gap_weight = 0.05 if use_reliable_timestep_gap else 0.10
    timestep_gap_base_weight = float(config.get("timestep_gap_weight", default_gap_weight))
    timestep_gap_hard_max = float(
        config.get("timestep_gap_hard_max", 0.30 if use_reliable_timestep_gap else float("inf"))
    )
    if use_timestep_gap_invariance:
        if not timestep_gap_partners or timestep_gap_anchor in timestep_gap_partners:
            raise ValueError("Timestep gap partners must be non-empty and exclude the anchor")
        if len(timestep_gap_partner_weights) != len(timestep_gap_partners):
            raise ValueError("timestep_gap_partner_weights must match timestep_gap_partner_timesteps")
        if any(weight < 0.0 for weight in timestep_gap_partner_weights) or sum(timestep_gap_partner_weights) <= 0.0:
            raise ValueError("timestep_gap_partner_weights must be non-negative with positive sum")
        if (
            timestep_gap_anchor < 0
            or timestep_gap_anchor >= int(config["num_steps"])
            or min(timestep_gap_partners) < 0
            or max(timestep_gap_partners) >= int(config["num_steps"])
        ):
            raise ValueError("Timestep gap anchor and partners must be valid scheduler timesteps")
        if timestep_gap_base_weight < 0.0:
            raise ValueError("timestep_gap_weight must be non-negative")
        if timestep_gap_hard_max <= float(config.get("timestep_gap_reliable_margin", 0.10)):
            raise ValueError("timestep_gap_hard_max must exceed timestep_gap_reliable_margin")
    agreement_triplet_weight = float(config.get("agreement_triplet_weight", 0.25))
    agreement_gap_tolerance = float(config.get("agreement_gap_tolerance", 0.05))
    if use_agreement_weighted_triplet and agreement_triplet_weight < 0.0:
        raise ValueError("agreement_triplet_weight must be non-negative")
    if use_agreement_weighted_triplet and agreement_gap_tolerance < 0.0:
        raise ValueError("agreement_gap_tolerance must be non-negative")
    default_adaptive_candidates = [25, 30, 40] if use_adaptive_evidence_timestep else [10, 25, 40]
    adaptive_timestep_candidates = [
        int(value) for value in config.get("adaptive_timestep_candidates", default_adaptive_candidates)
    ]
    if use_teacher_timestep or use_adaptive_evidence_timestep:
        if len(adaptive_timestep_candidates) < 2:
            raise ValueError("adaptive timestep candidates must contain at least two values")
        if len(set(adaptive_timestep_candidates)) != len(adaptive_timestep_candidates):
            raise ValueError("adaptive_timestep_candidates must be unique")
        if min(adaptive_timestep_candidates) < 0 or max(adaptive_timestep_candidates) >= int(config["num_steps"]):
            raise ValueError("adaptive_timestep_candidates must be valid scheduler timesteps")
    energy_distill_anchor_timestep = int(config.get("energy_distill_anchor_timestep", 25))
    energy_distill_teacher_timesteps = [
        int(value) for value in config.get("energy_distill_teacher_timesteps", [25, 30, 40])
    ]
    energy_distill_teacher_weights = [
        float(value) for value in config.get("energy_distill_teacher_weights", [0.5, 0.25, 0.25])
    ]
    if use_multitime_energy_distill:
        if len(energy_distill_teacher_timesteps) < 2:
            raise ValueError("energy distillation requires at least two teacher timesteps")
        if len(set(energy_distill_teacher_timesteps)) != len(energy_distill_teacher_timesteps):
            raise ValueError("energy-distill teacher timesteps must be unique")
        if len(energy_distill_teacher_weights) != len(energy_distill_teacher_timesteps):
            raise ValueError("energy-distill teacher weights must match teacher timesteps")
        if energy_distill_anchor_timestep not in energy_distill_teacher_timesteps:
            raise ValueError("energy-distill teacher timesteps must include its anchor timestep")
        if (
            energy_distill_anchor_timestep < 0
            or min(energy_distill_teacher_timesteps) < 0
            or max(energy_distill_teacher_timesteps) >= int(config["num_steps"])
        ):
            raise ValueError("energy-distill timesteps must be valid scheduler indices")
        if any(weight < 0.0 for weight in energy_distill_teacher_weights) or sum(energy_distill_teacher_weights) <= 0.0:
            raise ValueError("energy-distill teacher weights must be non-negative with positive sum")
    use_da_cnce_trust = loss_mode == "c2u_feat_da_cnce_trust"
    use_c2u_loss = loss_mode in {"c2u_feat", "c2u_feat_infonce", "c2u_feat_da_cnce_trust"}
    use_align_loss = loss_mode == "c2u_feat_align"
    primitive_context = None
    if use_primitive_multilabel:
        primitive_context = build_action_primitive_context(
            config,
            text_embed,
            seen_labels,
            device,
        )
    visual_ambiguity_weights = (
        build_visual_ambiguity_weights(config, seen_labels, device)
        if use_da_cnce_trust
        else None
    )
    c23_geometry_distance = None
    c24_reliability = None
    if use_ordinal_geometry:
        metric_path = config.get("c23_geometry_metric_path")
        if not metric_path:
            raise ValueError("C23 ordinal geometry requires c23_geometry_metric_path")
        c23_geometry_distance = load_c23_geometry_distance(metric_path, text_embed, device)
    if use_soft_ordinal_geometry:
        reliability_path = config.get("c24_reliability_path")
        if not reliability_path:
            raise ValueError("C24 soft ordinal geometry requires c24_reliability_path")
        c24_reliability = load_c24_reliability(
            reliability_path,
            class_count=int(text_embed.shape[0]),
            device=device,
            alpha=float(config.get("c24_reliability_alpha", 2.0)),
            beta=float(config.get("c24_reliability_beta", 2.0)),
        )
    episodic_proto_context = None
    if use_episodic_primitive_proto:
        episodic_proto_context = build_seen_visual_prototype_context(
            config,
            seen_labels,
            text_class_count=int(text_embed.shape[0]),
            device=device,
        )
    aux = None
    use_projected_infonce = bool(config.get("projected_infonce_weight", 0.0)) and use_c2u_loss
    needs_aux = use_align_loss or use_multipos_align or use_neutral_align or use_primitive_multilabel or (use_c2u_loss and (
        float(config.get("gen_con_weight", 0.0)) > 0.0
        or float(config.get("text_align_weight", 0.0)) > 0.0
        or use_projected_infonce
    ))
    if needs_aux:
        preserve_aux_rng = use_primitive_multilabel and bool(
            config.get(
                "primitive_preserve_aux_rng",
                config.get("primitive_detach_backbone", True),
            )
        )
        cpu_rng_state = torch.get_rng_state() if preserve_aux_rng else None
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if preserve_aux_rng and torch.cuda.is_available()
            else None
        )
        aux = C2UAuxHeads(
            hidden_size=int(config["hidden_size"]),
            text_size=int(text_embed.shape[-1]),
            feature_size=int(config.get("in_channels", 256)),
            proj_size=int(config.get("proj_size", 256)),
            fixed_text_space=bool(config.get("fixed_text_space", False)),
        ).to(device=device, dtype=dtype)
        if use_primitive_feature_distribution:
            if primitive_context is None:
                raise RuntimeError("C17 primitive context was not initialized")
            aux.configure_primitive_feature_distribution(
                primitive_size=int(primitive_context["class_targets"].shape[1]),
                feature_size=int(config.get("in_channels", 256)),
                hidden_size=int(config.get("primitive_feature_distribution_hidden", 256)),
            )
            aux.primitive_feature_distribution.to(device=device, dtype=dtype)
        if use_episodic_primitive_proto:
            if primitive_context is None:
                raise RuntimeError("C19 primitive context was not initialized")
            aux.configure_episodic_proto_generator(
                primitive_size=int(primitive_context["class_targets"].shape[1]),
                feature_size=int(config.get("in_channels", 256)),
                hidden_size=int(config.get("episodic_proto_hidden", 64)),
            )
            aux.episodic_proto_generator.to(device=device, dtype=dtype)
        if use_frozen_gallery_match:
            aux.configure_frozen_gallery_matcher(
                feature_size=int(config.get("in_channels", 256)),
                text_size=int(pooled_text_embed(text_embed).shape[-1]),
                proj_size=int(config.get("f1_match_proj_size", config.get("proj_size", 256))),
            )
            aux.frozen_gallery_feature_proj.to(device=device, dtype=dtype)
            aux.frozen_gallery_csv_proj.to(device=device, dtype=dtype)
            aux.frozen_gallery_llm_proj.to(device=device, dtype=dtype)
            aux.frozen_gallery_gate.to(device=device, dtype=dtype)
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    clean_tide = None
    clean_tide_teacher = None
    if use_clean_tide:
        if aux is None or primitive_context is None:
            raise RuntimeError("D5 CleanTIDE requires the C10 primitive context and auxiliary head")
        clean_tide_strata = config.get("clean_tide_strata", [[1, 16], [17, 33], [34, 49]])
        if not isinstance(clean_tide_strata, list) or len(clean_tide_strata) < 2:
            raise ValueError("clean_tide_strata must contain at least two [min, max] ranges")
        for stratum in clean_tide_strata:
            if not isinstance(stratum, (list, tuple)) or len(stratum) != 2:
                raise ValueError("Each clean_tide_strata item must be [min_timestep, max_timestep]")
            low, high = int(stratum[0]), int(stratum[1])
            if low < 0 or high < low or high >= int(config["num_steps"]):
                raise ValueError("clean_tide_strata contains an invalid scheduler timestep range")
        clean_tide = CleanTideNeutralBranch(
            feature_size=int(config.get("in_channels", 256)),
            hidden_size=int(config["hidden_size"]),
            feature_tokens=int(config.get("feature_tokens", 1)),
            strata=len(clean_tide_strata),
        ).to(device=device, dtype=dtype)
        if not bool(args.eval_only):
            teacher_checkpoint = config.get("clean_tide_teacher_checkpoint")
            if not teacher_checkpoint:
                raise ValueError("clean_tide_teacher_checkpoint is required for D5 training")
            teacher_path = repo_path(teacher_checkpoint)
            if not (teacher_path / "checkpoint.pt").exists():
                raise FileNotFoundError(f"D5 teacher checkpoint not found: {teacher_path}")
            clean_tide_teacher = ModelClass(**model_kwargs).to(device=device, dtype=dtype)
            teacher_metadata = load_checkpoint(teacher_path, clean_tide_teacher)
            clean_tide_teacher.eval().requires_grad_(False)
            print(
                "D5 CleanTIDE frozen teacher initialized from "
                f"{teacher_path} (epoch={teacher_metadata.get('epoch', 'unknown')}); "
                f"strata={clean_tide_strata}"
            )
    if use_clean_tide_multilayer:
        if aux is None or primitive_context is None:
            raise RuntimeError("D5b CleanTIDE requires the C10 primitive context and auxiliary head")
        if ModelClass is not DiTFineGrained:
            raise ValueError("D5b CleanTIDE currently requires model_type=fine_grained")
        clean_tide_layers = [int(value) for value in config.get("clean_tide_layers", [4, 8, 12])]
        if not clean_tide_layers or len(set(clean_tide_layers)) != len(clean_tide_layers):
            raise ValueError("clean_tide_layers must contain unique DiT layer indices")
        if min(clean_tide_layers) < 1 or max(clean_tide_layers) > int(config["depth"]):
            raise ValueError("clean_tide_layers must be within the primary DiT depth")
        clean_tide_buckets = build_logsnr_timestep_buckets(
            scheduler,
            num_buckets=int(config.get("clean_tide_logsnr_bins", 3)),
            start_timestep=int(config.get("clean_tide_timestep_min", 1)),
            end_timestep=int(config.get("clean_tide_timestep_max", int(config["num_steps"]) - 1)),
            device=device,
        )
        # D5b must leave the primary A0/C10 random trajectory unchanged.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        clean_tide = CleanTideMultiLayerStudent(
            feature_size=int(config.get("in_channels", 256)),
            hidden_size=int(config["hidden_size"]),
            feature_tokens=int(config.get("feature_tokens", 8)),
            num_heads=int(config["num_heads"]),
            depth=int(config["depth"]),
            alignment_layers=clean_tide_layers,
            strata=len(clean_tide_buckets),
            projection_multiplier=int(config.get("clean_tide_projection_multiplier", 2)),
            projection_depth=int(config.get("clean_tide_projection_depth", 3)),
        ).to(device=device, dtype=dtype)
        if not bool(args.eval_only):
            teacher_checkpoint = config.get("clean_tide_teacher_checkpoint")
            if not teacher_checkpoint:
                raise ValueError("clean_tide_teacher_checkpoint is required for D5b training")
            teacher_path = repo_path(teacher_checkpoint)
            if not (teacher_path / "checkpoint.pt").exists():
                raise FileNotFoundError(f"D5b teacher checkpoint not found: {teacher_path}")
            clean_tide_teacher = ModelClass(**model_kwargs).to(device=device, dtype=dtype)
            teacher_metadata = load_checkpoint(teacher_path, clean_tide_teacher)
            clean_tide_teacher.eval().requires_grad_(False)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            bucket_description = [bucket.detach().cpu().tolist() for bucket in clean_tide_buckets]
            print(
                "D5b CleanTIDE frozen teacher initialized from "
                f"{teacher_path} (epoch={teacher_metadata.get('epoch', 'unknown')}); "
                f"layers={clean_tide_layers} logsnr_buckets={bucket_description}"
            )

    global_step = 0
    start_epoch = 0
    best_acc = 0.0
    da_cnce_reference_model: DiT | None = None
    da_cnce_dual = float(config.get("da_cnce_dual_init", 0.0))
    early_stop_patience = max(0, int(config.get("early_stop_patience", 0)))
    max_train_batches = max(0, int(config.get("max_train_batches", 0)))
    epochs_without_improvement = 0
    use_ema = bool(config.get("use_ema", False))
    ema_decay = float(config.get("ema_decay", 0.999))
    timestep_teacher = None
    multitime_energy_teacher = None
    teacher_checkpoint = config.get("adaptive_timestep_teacher_checkpoint")
    if (use_teacher_timestep or use_timestep_policy_distill) and not bool(args.eval_only):
        if not teacher_checkpoint:
            raise ValueError(
                "adaptive_timestep_teacher_checkpoint is required for teacher/policy timestep training"
            )
        teacher_path = repo_path(teacher_checkpoint)
        if not (teacher_path / "checkpoint.pt").exists():
            raise FileNotFoundError(f"Adaptive timestep teacher checkpoint not found: {teacher_path}")
        if use_timestep_policy_distill:
            timestep_teacher = ModelClass(**model_kwargs).to(device=device, dtype=dtype)
            teacher_metadata = load_checkpoint(teacher_path, timestep_teacher)
            timestep_teacher.eval().requires_grad_(False)
        else:
            teacher_metadata = load_checkpoint(teacher_path, model)
            timestep_teacher = copy.deepcopy(model).eval().requires_grad_(False)
        print(
            "Timestep policy teacher initialized from "
            f"{teacher_path} (epoch={teacher_metadata.get('epoch', 'unknown')})"
        )
    if use_multitime_energy_distill and not bool(args.eval_only):
        energy_teacher_checkpoint = config.get("energy_distill_teacher_checkpoint")
        if not energy_teacher_checkpoint:
            raise ValueError("energy_distill_teacher_checkpoint is required for multi-timestep energy distillation")
        energy_teacher_path = repo_path(energy_teacher_checkpoint)
        if not (energy_teacher_path / "checkpoint.pt").exists():
            raise FileNotFoundError(f"Energy-distill teacher checkpoint not found: {energy_teacher_path}")
        multitime_energy_teacher = ModelClass(**model_kwargs).to(device=device, dtype=dtype)
        energy_teacher_metadata = load_checkpoint(energy_teacher_path, multitime_energy_teacher)
        multitime_energy_teacher.eval().requires_grad_(False)
        print(
            "MultiTimestepEnergyDistill frozen teacher initialized from "
            f"{energy_teacher_path} (epoch={energy_teacher_metadata.get('epoch', 'unknown')}); "
            f"anchor={energy_distill_anchor_timestep} views={energy_distill_teacher_timesteps}"
        )
    init_checkpoint = config.get("init_from_checkpoint")
    if init_checkpoint and not args.resume_from_checkpoint and not bool(args.eval_only):
        init_path = repo_path(init_checkpoint)
        if not (init_path / "checkpoint.pt").exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        init_metadata = load_checkpoint(init_path, model)
        print(
            f"Model weights initialized from {init_path} "
            f"(epoch={init_metadata.get('epoch', 'unknown')}); optimizer and scheduler reset"
        )

    lora_enabled = bool(config.get("lora_enabled", False))
    lora_targets: list[str] = []
    if lora_enabled:
        if ModelClass is not DiTFineGrained:
            raise ValueError("lora_enabled currently requires model_type=fine_grained")
        lora_targets = inject_finegrained_lora(
            model,
            rank=int(config.get("lora_rank", 4)),
            alpha=float(config.get("lora_alpha", config.get("lora_rank", 4))),
        )
        if not lora_targets:
            raise RuntimeError("No FineGrained DiT layers were selected for LoRA")

    model_trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    aux_trainable_params = (
        [parameter for parameter in aux.parameters() if parameter.requires_grad]
        if aux is not None
        else []
    )
    clean_tide_trainable_params = (
        [parameter for parameter in clean_tide.parameters() if parameter.requires_grad]
        if clean_tide is not None
        else []
    )
    trainable_params = model_trainable_params + aux_trainable_params + clean_tide_trainable_params
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after model freezing")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = int(config["num_iter"])
    lr_scheduler = get_scheduler(
        str(config.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(config.get("num_warmup", 0)),
        num_training_steps=total_steps,
    )
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if args.resume_from_checkpoint:
        metadata = load_checkpoint(
            repo_path(args.resume_from_checkpoint),
            model,
            None if args.eval_only else optimizer,
            None if args.eval_only else lr_scheduler,
            aux=aux,
            clean_tide=clean_tide,
            ema_model=ema_model,
        )
        global_step = int(metadata.get("global_step", 0))
        start_epoch = int(metadata.get("epoch", 0))
        best_acc = float(metadata.get("best_zsl_acc", 0.0))

    if bool(args.eval_only):
        if bool(config.get("eval_uot_probe", False)):
            if aux is None or primitive_context is None:
                raise RuntimeError("eval_uot_probe requires a C10 primitive checkpoint and context")
            uot_metrics = evaluate_uot_probe(
                model=model,
                aux=aux,
                loader=train_loader,
                scheduler=scheduler,
                text_embed=text_embed,
                seen_labels=seen_labels,
                primitive_context=primitive_context,
                config=config,
                device=device,
                dtype=dtype,
            )
            tag = args.eval_tag or "uot_probe"
            (work_dir / f"uot_probe_{tag}.json").write_text(
                json.dumps(uot_metrics, indent=2), encoding="utf-8"
            )
            print(json.dumps(uot_metrics, indent=2))
            return
        # Precision matrix must be built before eval_only early-return.
        _eval_precision = None
        if bool(config.get("eval_mahal", False)):
            _shrink = float(config.get("eval_mahal_shrinkage", config.get("dcr_shrinkage", 0.1)))
            _eval_precision = estimate_shared_precision(config, unseen_labels, _shrink, device)
            print(f"Mahalanobis eval: precision matrix shape={tuple(_eval_precision.shape)}")
        _bridge_ctx = build_seen_bridge_context(config, unseen_labels, text_embed, device)
        if _bridge_ctx is not None:
            print(f"Seen-class bridge: {_bridge_ctx.get('description', _bridge_ctx.get('mode', 'enabled'))}")
        _analogy_ctx = build_seen_analogy_context(config, unseen_labels, text_embed, device)
        eval_model = ema_model if ema_model is not None else model
        eval_aux = aux
        metrics = evaluate_zsl(eval_model, test_loader, scheduler, text_embed, unseen_labels, config, device, dtype, aux=eval_aux, precision=_eval_precision, bridge_ctx=_bridge_ctx, analogy_ctx=_analogy_ctx, primitive_ctx=primitive_context, episodic_proto_ctx=episodic_proto_context, clean_tide=clean_tide)
        print(
            json.dumps(
                {
                    "accuracy": metrics["accuracy"],
                    "base_accuracy": metrics.get("base_accuracy", metrics["accuracy"]),
                    "proj_accuracy": metrics.get("proj_accuracy", -1.0),
                    "proj_effective_alpha": metrics.get("proj_effective_alpha", 0.0),
                    "primitive_accuracy": metrics.get("primitive_accuracy", -1.0),
                    "primitive_effective_alpha": metrics.get("primitive_effective_alpha", 0.0),
                    "primitive_candidate_reliability": metrics.get("primitive_candidate_reliability", 1.0),
                    "clean_tide_primitive_accuracy": metrics.get("clean_tide_primitive_accuracy", -1.0),
                    "clean_tide_effective_alpha": metrics.get("clean_tide_effective_alpha", 0.0),
                    "feature_distribution_accuracy": metrics.get("feature_distribution_accuracy", -1.0),
                    "feature_distribution_effective_alpha": metrics.get("feature_distribution_effective_alpha", 0.0),
                    "episodic_proto_accuracy": metrics.get("episodic_proto_accuracy", -1.0),
                    "episodic_proto_seen_accuracy": metrics.get("episodic_proto_seen_accuracy", 0.0),
                    "episodic_proto_reliability": metrics.get("episodic_proto_reliability", 0.0),
                    "episodic_proto_effective_alpha": metrics.get("episodic_proto_effective_alpha", 0.0),
                    "hetero_multiview_enabled": metrics.get("hetero_multiview_enabled", False),
                    "hetero_timesteps": metrics.get("hetero_timesteps", []),
                    "hetero_mean_weights": metrics.get("hetero_mean_weights", []),
                    "hetero_mean_relative_variance": metrics.get("hetero_mean_relative_variance", []),
                    "hetero_correlation": metrics.get("hetero_correlation", []),
                    "hetero_negative_weight_fraction": metrics.get("hetero_negative_weight_fraction", 0.0),
                    "ot_enabled": metrics.get("ot_enabled", False),
                    "ot_epsilon": metrics.get("ot_epsilon", 0.0),
                    "ot_iterations": metrics.get("ot_iterations", 0),
                    "ot_marginal_error": metrics.get("ot_marginal_error", 0.0),
                    "semi_uot_enabled": metrics.get("semi_uot_enabled", False),
                    "semi_uot_epsilon": metrics.get("semi_uot_epsilon", 0.0),
                    "semi_uot_target_tau": metrics.get("semi_uot_target_tau", 0.0),
                    "semi_uot_iterations": metrics.get("semi_uot_iterations", 0),
                    "semi_uot_source_marginal_error": metrics.get(
                        "semi_uot_source_marginal_error", 0.0
                    ),
                    "semi_uot_target_uniform_kl": metrics.get(
                        "semi_uot_target_uniform_kl", 0.0
                    ),
                    "num_samples": metrics["num_samples"],
                    "samples_per_sec": metrics["samples_per_sec"],
                    "eval_timesteps": metrics["eval_timesteps"],
                    "eval_reverse_x0_enabled": metrics.get("eval_reverse_x0_enabled", False),
                    "eval_reverse_x0_timesteps": metrics.get("eval_reverse_x0_timesteps", []),
                    "eval_num_noise": metrics["eval_num_noise"],
                    "eval_noise_seed": metrics["eval_noise_seed"],
                    "eval_noise_mode": metrics["eval_noise_mode"],
                    "eval_distance_space": metrics["eval_distance_space"],
                    "bridge_mode": metrics.get("bridge_mode"),
                    "bridge_alpha": metrics.get("bridge_alpha", 0.0),
                    "bridge_effective_alpha": metrics.get("bridge_effective_alpha", 0.0),
                    "bridge_accuracy": metrics.get("bridge_accuracy", -1.0),
                    "bridge_fusion_stage": metrics.get("bridge_fusion_stage"),
                    "bridge_reliability": metrics.get("bridge_reliability", 0.0),
                    "use_ema": use_ema,
                    "checkpoint": args.resume_from_checkpoint,
                },
                indent=2,
            )
        )
        save_eval_artifacts(work_dir, metrics, args.eval_tag)
        gate_data = metrics.get("primitive_gate_data")
        if gate_data is not None:
            np.savez_compressed(work_dir / f"primitive_gate_{args.eval_tag}.npz", **gate_data)
        utility_gate_data = metrics.get("primitive_utility_gate_data")
        if utility_gate_data is not None:
            np.savez_compressed(
                work_dir / f"primitive_utility_gate_{args.eval_tag}.npz",
                **utility_gate_data,
            )
        eval_summary = {
            "accuracy": metrics["accuracy"],
            "base_accuracy": metrics.get("base_accuracy", metrics["accuracy"]),
            "primitive_accuracy": metrics.get("primitive_accuracy", -1.0),
            "primitive_effective_alpha": metrics.get("primitive_effective_alpha", 0.0),
            "primitive_candidate_reliability": metrics.get(
                "primitive_candidate_reliability", 1.0
            ),
            "clean_tide_primitive_accuracy": metrics.get("clean_tide_primitive_accuracy", -1.0),
            "clean_tide_effective_alpha": metrics.get("clean_tide_effective_alpha", 0.0),
            "primitive_risk_gate_mean": metrics.get("primitive_risk_gate_mean", 1.0),
            "primitive_utility_gate_enabled": metrics.get(
                "primitive_utility_gate_enabled", 1.0
            ),
            "primitive_score_mode": str(config.get("eval_primitive_score_mode", "c10")),
            "primitive_pu_positive_probability": float(
                config.get("primitive_pu_positive_probability", 0.95)
            ),
            "checkpoint": args.resume_from_checkpoint,
            "eval_tag": args.eval_tag,
            "num_samples": metrics["num_samples"],
            "eval_reverse_x0_enabled": metrics.get("eval_reverse_x0_enabled", False),
            "eval_reverse_x0_timesteps": metrics.get("eval_reverse_x0_timesteps", []),
        }
        (work_dir / f"metrics_{args.eval_tag}.json").write_text(
            json.dumps(eval_summary, indent=2), encoding="utf-8"
        )
        return

    train_log = (work_dir / "train_log.txt").open("a", encoding="utf-8")
    val_log = (work_dir / "val_log.txt").open("a", encoding="utf-8")

    def log(handle, text: str) -> None:
        print(text)
        handle.write(text + "\n")
        handle.flush()

    min_best_eval_samples = int(config.get("min_best_eval_samples", 0))
    eval_during_training = bool(config.get("eval_during_training", True))
    eval_split_name = str(config.get("eval_split_name", "Test"))
    grad_accum_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    total_epoch = int(config.get("num_epoch", 0)) or (
        total_steps * grad_accum_steps // max(len(train_loader), 1) + 1
    )
    loss_aware_second_moment = torch.ones(
        int(config["num_steps"]), device=device, dtype=torch.float32
    )
    loss_aware_observation_count = torch.zeros(
        int(config["num_steps"]), device=device, dtype=torch.long
    )
    easy_bank, hard_bank = load_negative_bank(config, device)
    topk_hard_bank = load_topk_hard_bank(config, device)
    if topk_hard_bank is not None:
        k = topk_hard_bank.shape[1]
        print(f"TopK hard neg bank loaded: shape={tuple(topk_hard_bank.shape)}, K={k} (seen-only negatives)")
    semantic_neighbor_bank = None
    semantic_text_features = None
    if use_semantic_distill:
        semantic_random_neg = int(config.get("semantic_random_neg", 1))
        if semantic_random_neg not in {0, 1}:
            raise ValueError(
                "B2 reuses A0's single random triplet negative, so semantic_random_neg must be 0 or 1"
            )
        semantic_neighbor_bank, semantic_text_features = build_semantic_neighbor_bank(
            pooled_text_embed(text_embed),
            seen_labels,
            int(config.get("semantic_topk", 2)),
        )
    ambiguity_neighbor_bank = None
    ambiguity_neighbor_description = None
    if use_ambiguity_contrast or use_natural_ambiguity:
        ambiguity_neighbor_bank, ambiguity_neighbor_description = build_ambiguity_neighbor_bank(
            config,
            text_embed,
            seen_labels,
        )
    prototype_direction_context = None
    if use_prototype_direction or use_adaptive_margin:
        prototype_direction_context = build_prototype_direction_context(
            config,
            text_embed,
            seen_labels,
            device,
        )
    align_layer = int(config.get("align_layer", config.get("depth", 8)))
    force_hard_pairs = load_force_hard_pairs(config)
    prediction_type = str(config.get("prediction_type", "sample"))
    dcr_weight = float(config.get("dcr_weight", 0.0))
    dcr_precision = None
    if dcr_weight > 0.0:
        dcr_precision = estimate_shared_precision(
            config, unseen_labels, float(config.get("dcr_shrinkage", 0.1)), device
        )
    # Precision matrix for Mahalanobis evaluation (reuse dcr_precision if available)
    eval_precision = None
    if bool(config.get("eval_mahal", False)):
        if dcr_precision is not None:
            eval_precision = dcr_precision
        else:
            shrinkage = float(config.get("eval_mahal_shrinkage", config.get("dcr_shrinkage", 0.1)))
            eval_precision = estimate_shared_precision(config, unseen_labels, shrinkage, device)
        print(f"Mahalanobis eval: precision matrix shape={tuple(eval_precision.shape)}")
    eval_bridge_ctx = build_seen_bridge_context(config, unseen_labels, text_embed, device)
    if eval_bridge_ctx is not None:
        print(f"Seen-class bridge: {eval_bridge_ctx.get('description', eval_bridge_ctx.get('mode', 'enabled'))}")
    eval_analogy_ctx = build_seen_analogy_context(config, unseen_labels, text_embed, device)
    if bool(config.get("require_negative_bank", False)) and (easy_bank is None or hard_bank is None):
        raise FileNotFoundError(f"Missing negative bank for config negative_bank_dir={config.get('negative_bank_dir')}")

    if start_epoch == 0 and global_step == 0:
        log(
            train_log,
            (
                f"Dataset: train_samples={len(train_loader.dataset)}\t"
                f"train_batches={len(train_loader)}\ttest_batches={len(test_loader)}\t"
                f"train_classes={train_class_labels.tolist()}\t"
                f"eval_classes={unseen_labels.tolist()}"
            ),
        )
        log(
            train_log,
            (
                f"Semantic: mode={config.get('text_condition_mode', 'clip_token')}\t"
                f"source={config.get('clip_text_source', 'concat')}\t"
                f"feature={config.get('semantic_feature_file', 'concat.npy')}\t"
                f"text_shape={tuple(text_embed.shape)}\ttext_dim={int(text_embed.shape[-1])}\tunseen={unseen_labels.tolist()}"
            ),
        )
        log(
            train_log,
            (
                f"Model: type={model_type}\t"
                f"class={ModelClass.__name__}\t"
                f"feature_tokens={int(config.get('feature_tokens', 1)) if ModelClass is DiTFineGrained else 1}\t"
                f"text_fusion={config.get('text_fusion', 'channel')}\t"
                f"text_mask={bool(config.get('mask_text_padding', False))}\t"
                f"self_attention={bool(config.get('enable_self_attention', True))}\t"
                f"cross_attention={bool(config.get('enable_cross_attention', True))}\t"
                f"global_text={bool(config.get('enable_global_text', True))}\t"
                f"text_injection_layers={config.get('text_injection_layers', 'all')}\t"
                f"text_injection_final={bool(config.get('text_injection_final', True))}\t"
                f"detail_layers={config.get('text_detail_layers', [])}\t"
                f"agreement_topk={config.get('agreement_topk', 0)}"
            ),
        )
        log(
            train_log,
            (
                f"Train: micro_batch={int(config['batch_size'])}\t"
                f"grad_accum={grad_accum_steps}\t"
                f"effective_batch={int(config['batch_size']) * grad_accum_steps}"
            ),
        )
        if lora_enabled:
            trainable_count = sum(parameter.numel() for parameter in trainable_params)
            total_count = sum(parameter.numel() for parameter in model.parameters())
            log(
                train_log,
                (
                    f"LoRA: rank={int(config.get('lora_rank', 4))}\t"
                    f"alpha={float(config.get('lora_alpha', config.get('lora_rank', 4))):g}\t"
                    f"layers={len(lora_targets)}\ttrainable={trainable_count}\t"
                    f"model_params={total_count}\tratio={trainable_count / max(1, total_count):.6f}"
                ),
            )
        if use_counterfactual_text:
            masked_classes = sum(
                bool(positions) for positions in primitive_context["counterfactual_positions"]
            ) if primitive_context is not None else 0
            log(
                train_log,
                (
                    f"CounterfactualTextEnergy: csv_high_idf_mask={masked_classes}/{int(text_embed.shape[0])}\t"
                    f"margin={float(config.get('counterfactual_text_margin', 0.02)):.4f}\t"
                    f"weight={float(config.get('counterfactual_text_weight', 0.05)):.4f}\t"
                    f"warmup={int(config.get('primitive_warmup_steps', 1000))}\t"
                    "same_noise=True\tsame_timestep=True\twrong_class_negative=False"
                ),
            )
        if init_checkpoint:
            log(
                train_log,
                f"Initialization: checkpoint={init_checkpoint}\toptimizer_reset=True\tscheduler_reset=True",
            )
        if min_best_eval_samples > 0:
            log(train_log, f"Best checkpoint requires eval_samples >= {min_best_eval_samples}")
        log(
            train_log,
            (
                "TextSim unseen before: "
                f"min={before_text_stats['min']:.4f}\tmean={before_text_stats['mean']:.4f}\tmax={before_text_stats['max']:.4f}"
            ),
        )
        log(
            train_log,
            (
                "TextSim unseen after: "
                f"min={after_text_stats['min']:.4f}\tmean={after_text_stats['mean']:.4f}\tmax={after_text_stats['max']:.4f}\t"
                f"transform={text_transform_info or {'mode': 'none'}}"
            ),
        )
        if use_projected_infonce:
            log(
                train_log,
                (
                    f"ProjectedInfoNCE: weight={float(config.get('projected_infonce_weight', 0.0))}\t"
                    f"tau={float(config.get('projected_infonce_tau', config.get('infonce_tau', 0.1)))}"
                ),
            )
        if use_semantic_distill:
            log(
                train_log,
                (
                    f"SemanticTopology: weight={float(config.get('semantic_distill_weight', 0.1))}\t"
                    f"warmup={int(config.get('semantic_distill_warmup_steps', 1000))}\t"
                    f"decay_end={int(config.get('semantic_distill_decay_end_steps', 0))}\t"
                    f"teacher_tau={float(config.get('semantic_teacher_tau', 0.1))}\t"
                    f"student_tau={float(config.get('semantic_student_tau', 0.05))}\t"
                    f"topk={int(config.get('semantic_topk', 2))}\t"
                    f"random={int(config.get('semantic_random_neg', 1))}\t"
                    "source=current_concat_condition"
                ),
            )
        if use_prototype_direction:
            log(
                train_log,
                (
                    f"PrototypeDirection: {prototype_direction_context['description']}\t"
                    f"weight={float(config.get('prototype_direction_weight', 0.01))}\t"
                    f"warmup={int(config.get('prototype_direction_warmup_steps', 1000))}"
                ),
            )
        if use_adaptive_margin:
            log(
                train_log,
                (
                    f"AdaptiveMargin: {prototype_direction_context['description']}\t"
                    f"base={float(config.get('tdsm_triplet_margin', 0.1)):.4f}\t"
                    f"min={float(config.get('adaptive_margin_min', 0.05)):.4f}\t"
                    f"max={float(config.get('adaptive_margin_max', 0.15)):.4f}"
                ),
            )
        if use_noise_consistent:
            log(
                train_log,
                (
                    f"NoiseConsistentEnergy: views={noise_consistent_views}\t"
                    f"beta={noise_consistent_beta:.4f}\t"
                    f"margin={float(config.get('tdsm_triplet_margin', 0.1)):.4f}\t"
                    "shared_timestep=True\trandom_negative=A0"
                ),
            )
        if use_timestep_gap_invariance:
            log(
                train_log,
                (
                    ("ReliableTimestepGap: " if use_reliable_timestep_gap else "ScaleCalibratedTimestepGap: ")
                    +
                    f"anchor={timestep_gap_anchor}\tpartners={timestep_gap_partners}\t"
                    f"partner_weights={timestep_gap_partner_weights}\t"
                    f"reliable_margin={float(config.get('timestep_gap_reliable_margin', 0.10)):.4f}\t"
                    f"hard_max={timestep_gap_hard_max:.4f}\t"
                    f"floor={float(config.get('timestep_gap_partner_floor', 0.0)):.4f}\t"
                    f"huber_delta={float(config.get('timestep_gap_huber_delta', 0.10)):.4f}\t"
                    f"weight={timestep_gap_base_weight:.4f}\t"
                    f"start={int(config.get('timestep_gap_start_step', 800))}\t"
                    f"warmup={int(config.get('timestep_gap_warmup_steps', 800))}\t"
                    "anchor_detached=True\tsame_epsilon=True"
                ),
            )
        if use_loss_aware_timestep:
            log(
                train_log,
                (
                    "LossAwareTimestepSampler: "
                    f"start={loss_aware_start_step}\t"
                    f"ema_decay={loss_aware_ema_decay:.4f}\t"
                    f"uniform_mix={loss_aware_uniform_mix:.3f}\t"
                    "stat=per_sample_recon_plus_random_rank_sq\t"
                    "importance=1/(T*p_t)\tobjective=uniform_t"
                ),
            )
        if use_teacher_timestep:
            log(
                train_log,
                (
                    f"TeacherAdaptiveTimestep: checkpoint={teacher_checkpoint}\t"
                    f"candidates={adaptive_timestep_candidates}\t"
                    f"anchor={int(config.get('adaptive_timestep_anchor', 25))}\t"
                    f"anchor_prob={float(config.get('adaptive_timestep_anchor_probability', 0.5)):.3f}\t"
                    f"temperature={float(config.get('adaptive_timestep_temperature', 0.05)):.4f}\t"
                    "negative=A0_random"
                ),
            )
        if use_timestep_policy_distill:
            log(
                train_log,
                (
                    "TimestepPolicyDistill: "
                    f"teacher={teacher_checkpoint}\t"
                    f"candidates={adaptive_timestep_candidates}\t"
                    f"noise={int(config.get('timestep_policy_num_noise', 3))}\t"
                    f"temperature={float(config.get('timestep_policy_temperature', 0.50)):.4f}\t"
                    f"uniform_floor={float(config.get('timestep_policy_uniform_floor', 0.10)):.4f}\t"
                    "evidence=top2_margin/noise_std\timportance=uniform_t"
                ),
            )
        elif use_adaptive_evidence_timestep:
            log(
                train_log,
                (
                    "AdaptiveEvidenceTimestep: "
                    f"candidates={adaptive_timestep_candidates}\t"
                    f"anchor={int(config.get('adaptive_timestep_anchor', 25))}\t"
                    f"temperature={float(config.get('adaptive_timestep_temperature', 0.10)):.4f}\t"
                    f"uniform_floor={float(config.get('adaptive_timestep_uniform_floor', 0.10)):.4f}\t"
                    "scout=current_model_seen_positive_negative	importance=uniform_t"
                ),
            )
        if use_multitime_energy_distill:
            log(
                train_log,
                (
                    "MultiTimestepEnergyDistill: "
                    f"teacher={config.get('energy_distill_teacher_checkpoint')}\t"
                    f"anchor={energy_distill_anchor_timestep}\t"
                    f"views={energy_distill_teacher_timesteps}\t"
                    f"weights={energy_distill_teacher_weights}\t"
                    f"candidates=positive+{int(config.get('energy_distill_num_neg', 2))}_seen_negatives\t"
                    f"kl_weight={float(config.get('energy_distill_kl_weight', 0.05)):.4f}\t"
                    f"gap_weight={float(config.get('energy_distill_gap_weight', 0.02)):.4f}\t"
                    f"warmup={int(config.get('energy_distill_warmup_steps', 1000))}\t"
                    f"agreement_gate={bool(config.get('energy_distill_agreement_gate', True))}\t"
                    "shared_epsilon=True\tseen_only=True"
                ),
            )
        if use_ambiguity_contrast:
            log(
                train_log,
                (
                    f"AmbiguityContrast: {ambiguity_neighbor_description}\t"
                    f"safe_margin={float(config.get('tdsm_triplet_margin', 0.1)):.4f}\t"
                    f"ambiguous_margin={float(config.get('ambiguity_margin', 0.0)):.4f}\t"
                    f"ambiguous_weight={float(config.get('ambiguity_rank_weight', 0.25)):.4f}\t"
                    "shared_noise=True\tshared_timestep=True\tseen_only=True"
                ),
            )
        if use_natural_ambiguity:
            log(
                train_log,
                (
                    f"NaturalAmbiguityMargin: {ambiguity_neighbor_description}\t"
                    f"base_margin={float(config.get('tdsm_triplet_margin', 0.1)):.4f}\t"
                    f"ambiguous_margin={float(config.get('ambiguity_margin', 0.05)):.4f}\t"
                    "sampling=A0_random\tshared_noise=True\tshared_timestep=True\tseen_only=True"
                ),
            )
        if use_dual_random:
            log(
                train_log,
                (
                    "DualRandomContrast: negatives=2\taggregate=mean\tdistinct=True\t"
                    f"margin={float(config.get('tdsm_triplet_margin', 0.1)):.4f}\t"
                    "shared_noise=True\tshared_timestep=True\tsemantic_mining=False"
                ),
            )
        if use_multipos_align:
            log(
                train_log,
                (
                    f"MultiPositiveAlign: layer={int(config.get('align_layer', 8))}\t"
                    f"proj={int(config.get('proj_size', 256))}\t"
                    f"tau={float(config.get('multipos_align_tau', 0.07)):.4f}\t"
                    f"weight={float(config.get('multipos_align_weight', 0.05)):.4f}\t"
                    f"warmup={int(config.get('multipos_align_warmup_steps', 1000))}\t"
                    f"snr_floor={float(config.get('multipos_snr_floor', 0.05)):.4f}\t"
                    "views=csv+llm\tbidirectional=True\tclass_balanced=True"
                ),
            )
        if use_neutral_align:
            log(
                train_log,
                (
                    f"NeutralDebiasedAlign: layer={int(config.get('align_layer', 8))}\t"
                    f"space={int(config.get('proj_size', int(text_embed.shape[-1])))}\t"
                    f"tau={float(config.get('neutral_align_tau', 0.07)):.4f}\t"
                    f"weight={float(config.get('neutral_align_weight', 0.05)):.4f}\t"
                    f"warmup={int(config.get('neutral_align_warmup_steps', 1000))}\t"
                    f"negative_floor={float(config.get('neutral_align_negative_floor', 0.1)):.3f}\t"
                    f"negative_power={float(config.get('neutral_align_negative_power', 1.0)):.3f}\t"
                    f"detach_backbone={bool(config.get('neutral_align_detach_backbone', True))}\t"
                    "condition=zero\ttext_space=fixed"
                ),
            )
        if use_primitive_multilabel:
            if primitive_context is None:
                raise RuntimeError("C10 primitive context was not initialized")
            log(
                train_log,
                (
                    f"ActionPrimitiveMultiLabel: {primitive_context['description']}\t"
                    f"latent_slots={int(config.get('feature_tokens', 8))}\t"
                    f"layer={int(config.get('align_layer', 8))}\t"
                    f"space={int(config.get('proj_size', 1024))}\t"
                    f"weight={float(config.get('primitive_multilabel_weight', 0.05)):.4f}\t"
                    f"gamma_neg={float(config.get('primitive_negative_gamma', 2.0)):.2f}\t"
                    f"detach_backbone={bool(config.get('primitive_detach_backbone', True))}\t"
                    f"gradient_mode={primitive_gradient_mode}\t"
                    f"pcgrad_max_ratio={primitive_pcgrad_max_ratio:.3f}\t"
                    "condition=zero\tfeature_source=frozen_global_256"
                ),
            )
        if use_clean_tide:
            log(
                train_log,
                (
                    f"D5CleanTIDE: layer={align_layer}\t"
                    f"strata={clean_tide_strata}\t"
                    f"distill_weight={float(config.get('clean_tide_distill_weight', 0.05)):.4f}\t"
                    f"primitive_weight={float(config.get('clean_tide_primitive_weight', 0.05)):.4f}\t"
                    f"warmup={int(config.get('clean_tide_warmup_steps', 1000))}\t"
                    "teacher=frozen_C10\tcondition=zero\tstudent=clean_feature_only"
                ),
            )
        if use_clean_tide_multilayer:
            bucket_description = [bucket.detach().cpu().tolist() for bucket in clean_tide_buckets]
            log(
                train_log,
                (
                    f"D5bCleanTIDE: layers={clean_tide_layers}\t"
                    f"logsnr_buckets={bucket_description}\t"
                    f"distill_weight={float(config.get('clean_tide_distill_weight', 0.05)):.4f}\t"
                    f"warmup={int(config.get('clean_tide_warmup_steps', 2000))}\t"
                    "teacher_noise=antithetic_pair\tprojection=3x_residual_ffn\t"
                    "condition=zero_gated_into_primary_DiT"
                ),
            )
        if use_primitive_nnpu:
            log(
                train_log,
                (
                    "PrimitiveNnPU: positives=prompt_tokens\t"
                    "unlabeled=all_seen_instances\t"
                    f"prior_floor={float(config.get('primitive_pu_prior_floor', 0.01)):.3f}\t"
                    f"prior_ceiling={float(config.get('primitive_pu_prior_ceiling', 0.99)):.3f}\t"
                    "negative_mask=disabled"
                ),
            )
        if use_primitive_semi_uot:
            log(
                train_log,
                (
                    "PrimitiveSemiUOT: source=DiT_motion_tokens_hard_marginal\t"
                    "target=true_class_primitives_KL_relaxed\t"
                    f"epsilon={float(config.get('primitive_uot_epsilon', 0.10)):.4f}\t"
                    f"target_tau={float(config.get('primitive_uot_target_tau', 0.20)):.4f}\t"
                    f"iters={int(config.get('primitive_uot_iterations', 20))}\t"
                    f"weight={float(config.get('primitive_uot_weight', 0.01)):.4f}\t"
                    f"warmup={int(config.get('primitive_uot_warmup_steps', 1000))}\t"
                    "backbone=active\tnegatives=none"
                ),
            )
        if use_band_elbo_snr:
            if elbo_schedule_metadata is None:
                raise RuntimeError("Band-tempered ELBO schedule was not initialized")
            band_text = ",".join(
                f"{band['start']}-{band['end']}:{band['mass']:.2f}"
                for band in elbo_schedule_metadata["bands"]
            )
            log(
                train_log,
                (
                    "BandTemperedELBO: objective=x0_conditional_energy\t"
                    f"t={elbo_schedule_metadata['start']}-{elbo_schedule_metadata['end']}\t"
                    f"bands={band_text}\t"
                    f"power={elbo_schedule_metadata['power']:.3f}\t"
                    f"raw_w={elbo_schedule_metadata['raw_weight_min']:.4g}-"
                    f"{elbo_schedule_metadata['raw_weight_max']:.4g}\t"
                    f"objective_w={elbo_schedule_metadata['objective_weight_min']:.4f}-"
                    f"{elbo_schedule_metadata['objective_weight_max']:.4f}\t"
                    "recon=band_ELBO\trank_and_C10=uniform_importance_corrected"
                ),
            )
        if use_primitive_feature_distribution:
            log(
                train_log,
                (
                    "PrimitiveFeatureDistribution: conditional=primitive_set\t"
                    f"hidden={int(config.get('primitive_feature_distribution_hidden', 256))}\t"
                    f"nll_weight={float(config.get('primitive_feature_distribution_nll_weight', 0.03)):.4f}\t"
                    f"recon_weight={float(config.get('primitive_feature_distribution_recon_weight', 0.02)):.4f}\t"
                    "negative_pairs=none\tbackbone=detached"
                ),
            )
        if use_episodic_primitive_proto:
            log(
                train_log,
                (
                    "EpisodicPrototypeCompletion: holdout=self_class\t"
                    f"support_classes={len(seen_labels) - 1}\t"
                    f"hidden={int(config.get('episodic_proto_hidden', 64))}\t"
                    f"temperature={float(config.get('episodic_proto_temperature', 0.2)):.3f}\t"
                    f"mse_weight={float(config.get('episodic_proto_mse_weight', 0.05)):.4f}\t"
                    f"nll_weight={float(config.get('episodic_proto_nll_weight', 0.01)):.4f}\t"
                    "negative_pairs=none\tbackbone=detached"
                ),
            )
        if use_episodic_gallery:
            log(
                train_log,
                (
                    "EpisodicGalleryRanking: "
                    f"classes={int(config.get('episodic_gallery_size', 12))}\t"
                    f"interval={int(config.get('episodic_gallery_interval', 4))}\t"
                    f"start={int(config.get('episodic_gallery_start_step', 1000))}\t"
                    f"temperature={float(config.get('episodic_gallery_temperature', 0.10)):.3f}\t"
                    f"weight={float(config.get('episodic_gallery_weight', 0.05)):.4f}\t"
                    "gallery=random_seen_class_set\tno_text_teacher"
                ),
            )
        if use_frozen_gallery_match:
            log(
                train_log,
                (
                    "F1FrozenGalleryMatch: input=skeleton256+frozen_csv_llm_text\t"
                    f"seen_gallery={len(seen_labels)}\t"
                    f"proj={int(config.get('f1_match_proj_size', config.get('proj_size', 256)))}\t"
                    f"temperature={float(config.get('f1_match_temperature', 0.10)):.3f}\t"
                    f"weight={float(config.get('f1_match_weight', 0.10)):.4f}\t"
                    f"warmup={int(config.get('f1_match_warmup_steps', 1000))}\t"
                    "views=csv_llm_gated\tfeatures_and_text_encoders=frozen"
                ),
            )
        if use_ema:
            log(train_log, f"EMA: enabled\tdecay={ema_decay}")
        log(
            train_log,
            f"Evaluation during training: {eval_during_training}\tsplit={eval_split_name}",
        )
        if (
            config.get("eval_noise_seed", None) is not None
            or config.get("eval_num_noise", None) is not None
            or config.get("eval_timesteps", None) is not None
        ):
            log(
                train_log,
                (
                    f"Eval: timesteps={config.get('eval_timesteps', [int(config['idx_inference_step'])])}\t"
                    f"num_noise={int(config.get('eval_num_noise', config.get('num_noise', 1)))}\t"
                    f"noise_seed={config.get('eval_noise_seed', None)}\t"
                    f"noise_mode={config.get('eval_noise_mode', 'per_sample')}\t"
                    f"distance_space={config.get('eval_distance_space', 'x0')}"
                ),
            )

    for epoch in range(start_epoch, total_epoch):
        should_early_stop = False
        model.train()
        if aux is not None:
            aux.train()
        if clean_tide is not None:
            clean_tide.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_easy_rank = 0.0
        epoch_hard_rank = 0.0
        epoch_infonce = 0.0
        epoch_proj_infonce = 0.0
        epoch_gen_con = 0.0
        epoch_text_align = 0.0
        epoch_dcr = 0.0
        epoch_semantic_kd = 0.0
        epoch_prototype_direction = 0.0
        epoch_prototype_reliability = 0.0
        epoch_adaptive_margin = 0.0
        epoch_margin_reliability = 0.0
        epoch_rank_active = 0.0
        epoch_noise_gap = 0.0
        epoch_noise_gap_std = 0.0
        epoch_rank_agreement = 0.0
        epoch_timestep_gap_loss = 0.0
        epoch_timestep_gap_coverage = 0.0
        epoch_timestep_gap_anchor = 0.0
        epoch_timestep_gap_partner = 0.0
        epoch_adaptive_timestep = 0.0
        epoch_teacher_gap = 0.0
        epoch_teacher_valid = 0.0
        epoch_ambiguity_rank = 0.0
        epoch_ambiguity_coverage = 0.0
        epoch_ambiguity_active = 0.0
        epoch_multipos_snr = 0.0
        epoch_neutral_negative_weight = 0.0
        epoch_primitive_density = 0.0
        epoch_clean_tide_distill = 0.0
        epoch_clean_tide_primitive = 0.0
        epoch_clean_tide_cosine = 0.0
        epoch_primitive_pu_active = 0.0
        epoch_primitive_pu_corrected_negative = 0.0
        epoch_primitive_proto_contrast = 0.0
        epoch_primitive_proto_negative_weight = 0.0
        epoch_primitive_tmr_infonce = 0.0
        epoch_primitive_tmr_filtered = 0.0
        epoch_primitive_feature_nll = 0.0
        epoch_primitive_feature_recon = 0.0
        epoch_primitive_feature_variance = 0.0
        epoch_episodic_proto_mse = 0.0
        epoch_episodic_proto_nll = 0.0
        epoch_episodic_gallery_rank = 0.0
        epoch_episodic_gallery_coverage = 0.0
        epoch_episodic_gallery_margin = 0.0
        epoch_f1_gallery_match = 0.0
        epoch_f1_csv_gate = 0.0
        epoch_counterfactual_text_loss = 0.0
        epoch_counterfactual_text_gap = 0.0
        epoch_energy_calibration = 0.0
        epoch_energy_distill_kl = 0.0
        epoch_energy_distill_gap = 0.0
        epoch_energy_distill_agreement = 0.0
        epoch_primitive_uot = 0.0
        epoch_primitive_uot_target_l1 = 0.0
        epoch_ordinal_geometry = 0.0
        epoch_ordinal_geometry_coverage = 0.0
        epoch_primitive_grad_cosine = 0.0
        epoch_primitive_grad_scale = 0.0
        epoch_primitive_grad_conflict = 0.0
        epoch_elbo_timestep_mean = 0.0
        epoch_elbo_uniform_importance = 0.0
        epoch_loss_aware_entropy = 0.0
        epoch_loss_aware_importance = 0.0
        epoch_loss_aware_active = 0.0
        epoch_timestep_counts = [0 for _ in adaptive_timestep_candidates]
        epoch_batches = 0
        seen_tensor_choices = [int(item) for item in seen_labels]
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        log(train_log, f"========= Epoch {epoch + 1} of {total_epoch} =========")

        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="train")):
            features = features.to(device=device, dtype=dtype).unsqueeze(1)
            labels = labels.to(device=device, dtype=torch.long)
            bsz = features.shape[0]
            noise = torch.randn_like(features)
            train_t_min = int(config.get("train_timestep_min", 0))
            train_t_max = int(config.get("train_timestep_max", int(config["num_steps"]) - 1))
            train_t_min = max(0, min(train_t_min, int(config["num_steps"]) - 1))
            train_t_max = max(train_t_min, min(train_t_max, int(config["num_steps"]) - 1))
            elbo_uniform_importance = torch.ones((bsz,), device=device, dtype=dtype)
            loss_aware_importance = torch.ones((bsz,), device=device, dtype=dtype)
            loss_aware_entropy = torch.tensor(0.0, device=device)
            loss_aware_active = 0.0
            if use_band_elbo_snr:
                if elbo_sampling_probabilities is None or elbo_objective_weights is None:
                    raise RuntimeError("Band-tempered ELBO sampler was not initialized")
                timesteps = torch.multinomial(elbo_sampling_probabilities, bsz, replacement=True).long()
                # Reconstruction is optimized under the ELBO-band density p(t).
                # The pre-existing ranking and C10 terms retain their original
                # uniform-timestep expectation through this importance factor.
                elbo_uniform_importance = elbo_objective_weights[timesteps].reciprocal().to(dtype=dtype)
            elif use_loss_aware_timestep and global_step >= loss_aware_start_step:
                eligible_moments = loss_aware_second_moment[train_t_min : train_t_max + 1]
                loss_aware_probabilities = loss_aware_timestep_probabilities(
                    eligible_moments,
                    loss_aware_uniform_mix,
                )
                sampled_positions = torch.multinomial(
                    loss_aware_probabilities,
                    bsz,
                    replacement=True,
                )
                timesteps = sampled_positions.add(train_t_min).long()
                loss_aware_importance = (
                    1.0
                    / (
                        float(loss_aware_probabilities.numel())
                        * loss_aware_probabilities[sampled_positions]
                    )
                ).to(dtype=dtype)
                loss_aware_entropy = -(
                    loss_aware_probabilities
                    * loss_aware_probabilities.clamp_min(1e-12).log()
                ).sum()
                loss_aware_active = 1.0
            else:
                timesteps = torch.randint(train_t_min, train_t_max + 1, (bsz,), device=device).long()
            if use_band_elbo_snr:
                epoch_elbo_timestep_mean += float(timesteps.float().mean().detach().cpu())
                epoch_elbo_uniform_importance += float(elbo_uniform_importance.mean().detach().cpu())
            noisy = scheduler.add_noise(features, noise, timesteps)
            target = diffusion_training_target(prediction_type, features, noise, timesteps, scheduler)
            fc, fl = split_text_condition(text_embed, labels)
            semantic_kd = torch.tensor(0.0, device=device)
            prototype_direction = torch.tensor(0.0, device=device)
            prototype_reliability = torch.tensor(0.0, device=device)
            adaptive_margin_mean = torch.tensor(0.0, device=device)
            adaptive_margin_reliability = torch.tensor(0.0, device=device)
            rank_active_fraction = torch.tensor(0.0, device=device)
            noise_gap_mean = torch.tensor(0.0, device=device)
            noise_gap_std = torch.tensor(0.0, device=device)
            rank_agreement_coverage = torch.tensor(0.0, device=device)
            timestep_gap_loss = torch.tensor(0.0, device=device)
            timestep_gap_coverage = torch.tensor(0.0, device=device)
            timestep_gap_anchor_mean = torch.tensor(0.0, device=device)
            timestep_gap_partner_mean = torch.tensor(0.0, device=device)
            timestep_gap_weight = 0.0
            adaptive_timestep_mean = torch.tensor(0.0, device=device)
            teacher_gap_mean = torch.tensor(0.0, device=device)
            teacher_valid_fraction = torch.tensor(0.0, device=device)
            ambiguity_rank_loss = torch.tensor(0.0, device=device)
            ambiguity_coverage = torch.tensor(0.0, device=device)
            ambiguity_active_fraction = torch.tensor(0.0, device=device)
            multipos_snr_mean = torch.tensor(0.0, device=device)
            neutral_negative_weight = torch.tensor(0.0, device=device)
            primitive_target_density = torch.tensor(0.0, device=device)
            clean_tide_distill = torch.tensor(0.0, device=device)
            clean_tide_primitive = torch.tensor(0.0, device=device)
            clean_tide_cosine = torch.tensor(0.0, device=device)
            primitive_pu_active = torch.tensor(0.0, device=device)
            primitive_pu_corrected_negative = torch.tensor(0.0, device=device)
            primitive_proto_contrast = torch.tensor(0.0, device=device)
            primitive_proto_negative_weight = torch.tensor(0.0, device=device)
            primitive_tmr_infonce = torch.tensor(0.0, device=device)
            primitive_tmr_filtered = torch.tensor(0.0, device=device)
            primitive_feature_nll = torch.tensor(0.0, device=device)
            primitive_feature_recon = torch.tensor(0.0, device=device)
            primitive_feature_variance = torch.tensor(0.0, device=device)
            episodic_proto_mse = torch.tensor(0.0, device=device)
            episodic_proto_nll = torch.tensor(0.0, device=device)
            episodic_gallery_rank = torch.tensor(0.0, device=device)
            episodic_gallery_coverage = torch.tensor(0.0, device=device)
            episodic_gallery_margin = torch.tensor(0.0, device=device)
            episodic_gallery_weight = 0.0
            f1_gallery_match = torch.tensor(0.0, device=device)
            f1_csv_gate = torch.tensor(0.0, device=device)
            f1_gallery_match_weight = 0.0
            counterfactual_text_loss = torch.tensor(0.0, device=device)
            counterfactual_text_gap = torch.tensor(0.0, device=device)
            energy_calibration = torch.tensor(0.0, device=device)
            energy_distill_kl = torch.tensor(0.0, device=device)
            energy_distill_gap = torch.tensor(0.0, device=device)
            energy_distill_agreement = torch.tensor(0.0, device=device)
            energy_distill_weight = 0.0
            energy_distill_gap_weight = 0.0
            primitive_uot = torch.tensor(0.0, device=device)
            primitive_uot_target_l1 = torch.tensor(0.0, device=device)
            ordinal_geometry = torch.tensor(0.0, device=device)
            ordinal_geometry_coverage = torch.tensor(0.0, device=device)
            primitive_grad_cosine = 0.0
            primitive_grad_scale = 0.0
            primitive_grad_conflict = 0.0
            primitive_main_loss = None
            primitive_weighted_loss = None

            if loss_mode in {"c2u_feat", "c2u_feat_infonce", "c2u_feat_da_cnce_trust"}:
                if (
                    use_da_cnce_trust
                    and da_cnce_reference_model is None
                    and global_step >= int(config.get("da_cnce_trust_warmup_steps", 1000))
                ):
                    da_cnce_reference_model = copy.deepcopy(model).eval().requires_grad_(False)
                need_hidden = aux is not None and float(config.get("text_align_weight", 0.0)) > 0.0
                if need_hidden:
                    pred, hidden_states = model(
                        noisy,
                        timesteps.to(dtype),
                        fc,
                        fl,
                        return_hidden=True,
                        hidden_layers=[align_layer],
                    )
                    anchor_hidden = DiT.pool_hidden(hidden_states[0])
                else:
                    pred = model(noisy, timesteps.to(dtype), fc, fl)
                    anchor_hidden = None

                pred_easy = None
                pred_hard = None
                if easy_bank is not None and hard_bank is not None:
                    easy_labels = easy_bank[labels]
                    hard_labels = hard_bank[labels]
                else:
                    easy_labels = torch.tensor(random.choices(seen_tensor_choices, k=bsz), device=device, dtype=torch.long)
                    hard_labels = torch.tensor(random.choices(seen_tensor_choices, k=bsz), device=device, dtype=torch.long)
                    easy_labels = torch.where(easy_labels == labels, torch.roll(easy_labels, shifts=1), easy_labels)
                    hard_labels = torch.where(hard_labels == labels, torch.roll(hard_labels, shifts=-1), hard_labels)

                easy_fc, easy_fl = split_text_condition(text_embed, easy_labels)
                hard_fc, hard_fl = split_text_condition(text_embed, hard_labels)

                pred_sample = prediction_to_sample(pred, noisy, timesteps, scheduler, prediction_type)
                pos_dist = reconstruction_distance(pred_sample, features)
                recon = F.mse_loss(pred, target)
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                infonce = torch.tensor(0.0, device=device)
                projected_infonce = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                trust_region = torch.tensor(0.0, device=device)
                ordinal_geometry = torch.tensor(0.0, device=device)
                ordinal_geometry_coverage = torch.tensor(0.0, device=device)

                needs_pairwise = (
                    loss_mode == "c2u_feat"
                    or float(config.get("easy_weight", 0.0)) > 0.0
                    or float(config.get("hard_weight", 0.0)) > 0.0
                    or float(config.get("gen_con_weight", 0.0)) > 0.0
                    or dcr_weight > 0.0
                )
                if needs_pairwise:
                    pred_easy = model(noisy, timesteps.to(dtype), easy_fc, easy_fl)
                    pred_hard = model(noisy, timesteps.to(dtype), hard_fc, hard_fl)
                    pred_easy_sample = prediction_to_sample(pred_easy, noisy, timesteps, scheduler, prediction_type)
                    pred_hard_sample = prediction_to_sample(pred_hard, noisy, timesteps, scheduler, prediction_type)
                    easy_dist = reconstruction_distance(pred_easy_sample, features)
                    hard_dist = reconstruction_distance(pred_hard_sample, features)
                    easy_rank = torch.clamp(
                        pos_dist - easy_dist + float(config.get("margin_easy", config.get("margin", 0.1))),
                        min=0.0,
                    ).mean()
                    hard_rank = torch.clamp(
                        pos_dist - hard_dist + float(config.get("margin_hard", config.get("margin", 0.1))),
                        min=0.0,
                    ).mean()

                if dcr_weight > 0.0 and pred_hard is not None and dcr_precision is not None:
                    dcr = discriminative_counterfactual_loss(
                        pred_sample,
                        pred_hard_sample,
                        features,
                        dcr_precision,
                        margin=float(config.get("margin_dcr", config.get("margin_hard", 0.2))),
                    )

                if loss_mode in {"c2u_feat_infonce", "c2u_feat_da_cnce_trust"}:
                    negative_labels = sample_infonce_negative_labels(
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        num_neg=int(config.get("infonce_num_neg", 7)),
                        device=device,
                        hard_bank=hard_bank,
                        topk_hard_bank=topk_hard_bank,
                        force_hard_pairs=force_hard_pairs,
                    )
                    negative_weights = None
                    if use_da_cnce_trust:
                        if visual_ambiguity_weights is None:
                            raise RuntimeError("C30 visual ambiguity weights were not initialized")
                        negative_weights = visual_ambiguity_weights[labels[:, None], negative_labels].to(dtype=features.dtype)
                    infonce, _distance_matrix = reconstruction_infonce_loss(
                        model=model,
                        noisy=noisy,
                        timesteps=timesteps.to(dtype),
                        text_embed=text_embed,
                        labels=labels,
                        negative_labels=negative_labels,
                        target=features,
                        scheduler=scheduler,
                        prediction_type=prediction_type,
                        tau=float(config.get("infonce_tau", 0.1)),
                        negative_weights=negative_weights,
                    )
                    if use_projected_infonce and aux is not None:
                        projected_infonce = projected_reconstruction_infonce_loss(
                            model=model,
                            aux=aux,
                            noisy=noisy,
                            timesteps=timesteps.to(dtype),
                            text_embed=text_embed,
                            labels=labels,
                            negative_labels=negative_labels,
                            scheduler=scheduler,
                            prediction_type=prediction_type,
                            tau=float(config.get("projected_infonce_tau", config.get("infonce_tau", 0.1))),
                        )

                if use_da_cnce_trust and da_cnce_reference_model is not None:
                    with torch.no_grad():
                        reference_pred = da_cnce_reference_model(noisy, timesteps.to(dtype), fc, fl)
                    trust_region = F.mse_loss(pred, reference_pred)

                if aux is not None and float(config.get("gen_con_weight", 0.0)) > 0.0:
                    gen_con = generated_contrast_loss(
                        aux,
                        features.flatten(1),
                        pred_sample,
                        pred_easy_sample,
                        pred_hard_sample,
                        tau=float(config.get("gen_con_tau", 0.2)),
                    )
                if aux is not None and float(config.get("text_align_weight", 0.0)) > 0.0:
                    text_align = text_alignment_loss(
                        aux,
                        anchor_hidden,
                        fc,
                        labels,
                        tau=float(config.get("text_align_tau", 0.07)),
                    )
                loss = (
                    recon * float(config.get("rec_weight", 1.0))
                    + infonce * float(config.get("infonce_weight", 0.0))
                    + projected_infonce * float(config.get("projected_infonce_weight", 0.0))
                    + easy_rank * float(config.get("easy_weight", 0.25))
                    + hard_rank * float(config.get("hard_weight", 1.0))
                    + gen_con * float(config.get("gen_con_weight", 0.1))
                    + text_align * float(config.get("text_align_weight", 0.1))
                    + dcr * dcr_weight
                    + trust_region * da_cnce_dual
                )
            elif loss_mode in {"tide_x0_contrastive", "tide_x0_rank"}:
                pred = model(noisy, timesteps.to(dtype), fc, fl)
                pred_sample = prediction_to_sample(pred, noisy, timesteps, scheduler, prediction_type)
                pos_dist = reconstruction_distance(pred_sample, features)

                if topk_hard_bank is not None:
                    if bool(config.get("x0_hard_topk_random", True)):
                        hard_slots = torch.randint(
                            0,
                            int(topk_hard_bank.shape[1]),
                            (bsz,),
                            device=device,
                        )
                        hard_neg_labels = topk_hard_bank[labels, hard_slots]
                    else:
                        hard_slot = min(
                            max(0, int(config.get("tdsm_hard_topk_slot", 0))),
                            int(topk_hard_bank.shape[1]) - 1,
                        )
                        hard_neg_labels = topk_hard_bank[labels, hard_slot]
                elif hard_bank is not None:
                    hard_neg_labels = hard_bank[labels]
                else:
                    hard_neg_labels = sample_distinct_negative_labels(labels, seen_tensor_choices)

                random_neg_labels = sample_distinct_negative_labels(
                    labels,
                    seen_tensor_choices,
                    excluded_labels=hard_neg_labels,
                )
                candidate_negative_labels = (random_neg_labels, hard_neg_labels)
                candidate_distances = [pos_dist]
                for negative_labels in candidate_negative_labels:
                    negative_fc, negative_fl = split_text_condition(text_embed, negative_labels)
                    pred_negative = model(noisy, timesteps.to(dtype), negative_fc, negative_fl)
                    pred_negative_sample = prediction_to_sample(
                        pred_negative,
                        noisy,
                        timesteps,
                        scheduler,
                        prediction_type,
                    )
                    candidate_distances.append(
                        reconstruction_distance(pred_negative_sample, features)
                    )

                distance_matrix = torch.stack(candidate_distances, dim=1)
                contrastive_tau = max(float(config.get("x0_contrastive_tau", 0.05)), 1e-6)
                logits = -distance_matrix / contrastive_tau
                additive_margin = float(config.get("x0_contrastive_margin", 0.0))
                if additive_margin:
                    logits[:, 0] -= additive_margin / contrastive_tau
                contrastive_targets = torch.zeros((bsz,), device=device, dtype=torch.long)
                # Multiplying CE by tau keeps this term on the same scale as
                # reconstruction distances instead of letting log(K) dominate x0 MSE.
                infonce = F.cross_entropy(logits, contrastive_targets) * contrastive_tau
                warmup_steps = max(0, int(config.get("x0_contrastive_warmup_steps", 500)))
                warmup_scale = (
                    min(1.0, float(global_step + 1) / float(warmup_steps))
                    if warmup_steps > 0
                    else 1.0
                )
                contrastive_weight = float(config.get("x0_contrastive_weight", 0.5)) * warmup_scale

                recon = F.mse_loss(pred, target)
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = torch.tensor(0.0, device=device)
                projected_infonce = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                loss = (
                    recon * float(config.get("rec_weight", 1.0))
                    + infonce * contrastive_weight
                )
            elif loss_mode in {
                "tdsm_x0_triplet",
                "tdsm_x0",
                "x0_triplet",
                "tdsm_x0_hybrid_triplet",
                "tdsm_x0_hybrid",
                "tdsm_native_triplet",
                "tdsm_prediction_triplet",
                "tdsm_x0_semantic_distill",
                "tdsm_x0_topology",
                "tdsm_x0_proto_direction",
                "tdsm_x0_prototype_direction",
                "tdsm_x0_adaptive_margin",
                "tdsm_x0_geometry_margin",
                "tdsm_x0_noise_consistent",
                "tdsm_x0_noise_lcb",
                "tdsm_x0_agreement_weighted_triplet",
                "tdsm_x0_teacher_timestep",
                "tdsm_x0_adaptive_timestep",
                "tdsm_x0_action_primitive_adaptive_evidence",
                "tdsm_x0_action_primitive_timestep_policy_distill",
                "tdsm_x0_ambiguity_contrast",
                "tdsm_x0_false_negative_aware",
                "tdsm_x0_natural_ambiguity_margin",
                "tdsm_x0_dual_random_triplet",
                "tdsm_x0_multi_random_triplet",
                "tdsm_x0_multipos_align",
                "tdsm_x0_bidirectional_multipos",
                "tdsm_x0_neutral_debiased_align",
                "tdsm_x0_neutral_semantic_contrast",
                "tdsm_x0_action_primitive_multilabel",
                "tdsm_x0_latent_primitive_multilabel",
                "tdsm_x0_action_primitive_proto_contrast",
                "tdsm_x0_action_primitive_tmr_filtered",
                "tdsm_x0_action_primitive_feature_distribution",
                "tdsm_x0_action_primitive_episodic_proto",
                "tdsm_x0_action_primitive_episodic_gallery",
                "tdsm_x0_action_primitive_counterfactual_text",
                "tdsm_x0_action_primitive_energy_calibration",
                "tdsm_x0_action_primitive_ordinal_geometry",
                "tdsm_x0_action_primitive_ordinal_geometry_reliable",
                "tdsm_x0_action_primitive_soft_ordinal_geometry",
                "tdsm_x0_action_primitive_semi_uot",
                "tdsm_x0_action_primitive_nnpu",
                "tdsm_x0_action_primitive_band_elbo_snr",
                "tdsm_x0_action_primitive_timestep_gap",
        "tdsm_x0_action_primitive_timestep_reliable_gap",
        "tdsm_x0_action_primitive_adaptive_evidence",
        "tdsm_x0_action_primitive_timestep_policy_distill",
                "tdsm_x0_action_primitive_loss_aware_sampler",
                "tdsm_x0_action_primitive_multitime_energy_distill",
                "tdsm_x0_action_primitive_clean_tide",
                "tdsm_x0_action_primitive_clean_tide_multilayer",
                "tdsm_x0_action_primitive_frozen_gallery_match",
            }:
                use_native_triplet = loss_mode in {
                    "tdsm_native_triplet",
                    "tdsm_prediction_triplet",
                }
                ambiguous_neg_labels = None
                ambiguous_valid = None
                if use_ambiguity_contrast:
                    if ambiguity_neighbor_bank is None:
                        raise RuntimeError("Ambiguity neighbour bank was not initialized")
                    random_neg_labels, ambiguous_neg_labels, ambiguous_valid = (
                        sample_ambiguity_aware_negative_labels(
                            labels,
                            seen_tensor_choices,
                            ambiguity_neighbor_bank,
                        )
                    )
                elif use_dual_random:
                    random_neg_labels = sample_distinct_negative_labels(
                        labels,
                        seen_tensor_choices,
                    )
                else:
                    random_neg_labels = torch.tensor(
                        random.choices(seen_tensor_choices, k=bsz),
                        device=device,
                        dtype=torch.long,
                    )
                sampled_ambiguity_mask = None
                if use_natural_ambiguity:
                    if ambiguity_neighbor_bank is None:
                        raise RuntimeError("Ambiguity neighbour bank was not initialized")
                    positive_values = labels.detach().cpu().tolist()
                    negative_values = random_neg_labels.detach().cpu().tolist()
                    sampled_ambiguity_mask = torch.tensor(
                        [
                            int(negative) in ambiguity_neighbor_bank.get(int(positive), [])
                            for positive, negative in zip(positive_values, negative_values)
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
                random_fc, random_fl = split_text_condition(text_embed, random_neg_labels)
                random_valid = labels.ne(random_neg_labels)
                second_random_neg_labels = None
                second_random_valid = None
                if use_dual_random or use_ordinal_geometry or use_timestep_policy_distill:
                    second_random_neg_labels = sample_distinct_negative_labels(
                        labels,
                        seen_tensor_choices,
                        excluded_labels=random_neg_labels,
                    )
                    second_random_valid = labels.ne(second_random_neg_labels)

                if use_teacher_timestep or use_adaptive_evidence_timestep:
                    if (use_teacher_timestep or use_timestep_policy_distill) and timestep_teacher is None:
                        raise RuntimeError("Adaptive timestep teacher was not initialized")
                    scout_model = (
                        timestep_teacher
                        if (use_teacher_timestep or use_timestep_policy_distill)
                        else model
                    )
                    candidate_noisy = []
                    candidate_target = []
                    candidate_teacher_gaps = []
                    candidate_policy_margin_views = []
                    candidate_policy_energy_views = []
                    shared_candidate_noise = (
                        torch.randn_like(features)
                        if (use_adaptive_evidence_timestep and not use_timestep_policy_distill)
                        else None
                    )
                    with torch.no_grad():
                        for candidate_timestep in adaptive_timestep_candidates:
                            policy_noise_count = (
                                max(2, int(config.get("timestep_policy_num_noise", 3)))
                                if use_timestep_policy_distill
                                else 1
                            )
                            timestep_gap_views = []
                            timestep_margin_views = []
                            timestep_energy_views = []
                            for noise_index in range(policy_noise_count):
                                candidate_t = torch.full(
                                    (bsz,),
                                    int(candidate_timestep),
                                    device=device,
                                    dtype=torch.long,
                                )
                                candidate_noise = (
                                    shared_candidate_noise
                                    if shared_candidate_noise is not None
                                    else torch.randn_like(features)
                                )
                                candidate_x_t = scheduler.add_noise(features, candidate_noise, candidate_t)
                                candidate_training_target = diffusion_training_target(
                                    prediction_type,
                                    features,
                                    candidate_noise,
                                    candidate_t,
                                    scheduler,
                                )
                                if use_timestep_policy_distill:
                                    candidate_label_views = (
                                        (labels, random_neg_labels, second_random_neg_labels)
                                        if second_random_neg_labels is not None
                                        else (labels, random_neg_labels)
                                    )
                                    energy_views = []
                                    for candidate_labels_view in candidate_label_views:
                                        candidate_fc, candidate_fl = split_text_condition(
                                            text_embed, candidate_labels_view
                                        )
                                        candidate_pred = scout_model(
                                            candidate_x_t,
                                            candidate_t.to(dtype),
                                            candidate_fc,
                                            candidate_fl,
                                        )
                                        candidate_sample = prediction_to_sample(
                                            candidate_pred,
                                            candidate_x_t,
                                            candidate_t,
                                            scheduler,
                                            prediction_type,
                                        )
                                        energy_views.append(
                                            reconstruction_distance(candidate_sample, features)
                                        )
                                    candidate_energies = torch.stack(energy_views, dim=1)
                                    relative_energies = row_relative_energies(candidate_energies)
                                    sorted_energies = relative_energies.sort(dim=1).values
                                    timestep_margin_views.append(
                                        sorted_energies[:, 1] - sorted_energies[:, 0]
                                    )
                                    timestep_energy_views.append(candidate_energies)
                                    timestep_gap_views.append(
                                        candidate_energies[:, 1:].min(dim=1).values
                                        - candidate_energies[:, 0]
                                    )
                                else:
                                    teacher_positive = scout_model(
                                        candidate_x_t,
                                        candidate_t.to(dtype),
                                        fc,
                                        fl,
                                    )
                                    teacher_negative = scout_model(
                                        candidate_x_t,
                                        candidate_t.to(dtype),
                                        random_fc,
                                        random_fl,
                                    )
                                    teacher_positive_sample = prediction_to_sample(
                                        teacher_positive,
                                        candidate_x_t,
                                        candidate_t,
                                        scheduler,
                                        prediction_type,
                                    )
                                    teacher_negative_sample = prediction_to_sample(
                                        teacher_negative,
                                        candidate_x_t,
                                        candidate_t,
                                        scheduler,
                                        prediction_type,
                                    )
                                    timestep_gap_views.append(
                                        reconstruction_distance(teacher_negative_sample, features)
                                        - reconstruction_distance(teacher_positive_sample, features)
                                    )
                                if noise_index == 0:
                                    candidate_noisy.append(candidate_x_t)
                                    candidate_target.append(candidate_training_target)
                            gap_views = torch.stack(timestep_gap_views, dim=0)
                            candidate_teacher_gaps.append(gap_views.mean(dim=0))
                            if use_timestep_policy_distill:
                                candidate_policy_margin_views.append(
                                    torch.stack(timestep_margin_views, dim=0)
                                )
                                candidate_policy_energy_views.append(
                                    torch.stack(timestep_energy_views, dim=0)
                                )

                    teacher_gap_matrix = torch.stack(candidate_teacher_gaps)
                    candidate_timestep_tensor = torch.tensor(
                        adaptive_timestep_candidates,
                        device=device,
                        dtype=torch.long,
                    )
                    if use_timestep_policy_distill:
                        if not candidate_policy_margin_views or not candidate_policy_energy_views:
                            raise RuntimeError("Timestep policy distillation did not collect teacher evidence")
                        mean_margin = torch.stack(
                            [values.mean(dim=0) for values in candidate_policy_margin_views], dim=0
                        )
                        noise_std = torch.stack(
                            [
                                values.var(dim=0, unbiased=False).mean(dim=1).sqrt()
                                for values in candidate_policy_energy_views
                            ],
                            dim=0,
                        )
                        # This is the teacher version of evaluation evidence:
                        # candidate top-2 margin discounted by noise instability.
                        evidence = mean_margin / (
                            noise_std
                            + float(config.get("timestep_policy_variance_floor", 0.05))
                        )
                        temperature = max(
                            float(config.get("timestep_policy_temperature", 0.50)),
                            1e-6,
                        )
                        selection_probabilities = torch.softmax(evidence / temperature, dim=0)
                        uniform_floor = min(
                            max(float(config.get("timestep_policy_uniform_floor", 0.10)), 0.0),
                            1.0,
                        )
                        selection_probabilities = (
                            (1.0 - uniform_floor) * selection_probabilities
                            + uniform_floor / float(len(adaptive_timestep_candidates))
                        )
                        selected_timestep_index = torch.multinomial(
                            selection_probabilities.transpose(0, 1), 1
                        ).squeeze(1)
                        loss_aware_importance = (
                            1.0
                            / (
                                float(len(adaptive_timestep_candidates))
                                * selection_probabilities[
                                    selected_timestep_index,
                                    torch.arange(bsz, device=device),
                                ]
                            )
                        ).to(dtype=dtype)
                        loss_aware_active = 1.0
                        loss_aware_entropy = -(
                            selection_probabilities
                            * selection_probabilities.clamp_min(1e-12).log()
                        ).sum(dim=0).mean()
                    elif use_adaptive_evidence_timestep:
                        # Select high-evidence views, while retaining a uniform
                        # floor and correcting the sampled objective back to the
                        # uniform timestep expectation.
                        temperature = max(
                            float(config.get("adaptive_timestep_temperature", 0.10)),
                            1e-6,
                        )
                        candidate_scores = teacher_gap_matrix.detach().clamp_min(0.0)
                        selection_probabilities = torch.softmax(
                            candidate_scores / temperature,
                            dim=0,
                        )
                        uniform_floor = min(
                            max(float(config.get("adaptive_timestep_uniform_floor", 0.10)), 0.0),
                            1.0,
                        )
                        selection_probabilities = (
                            (1.0 - uniform_floor) * selection_probabilities
                            + uniform_floor / float(len(adaptive_timestep_candidates))
                        )
                        selected_timestep_index = torch.multinomial(
                            selection_probabilities.transpose(0, 1),
                            1,
                        ).squeeze(1)
                        loss_aware_importance = (
                            1.0
                            / (
                                float(len(adaptive_timestep_candidates))
                                * selection_probabilities[
                                    selected_timestep_index,
                                    torch.arange(bsz, device=device),
                                ]
                            )
                        ).to(dtype=dtype)
                        loss_aware_active = 1.0
                        loss_aware_entropy = -(
                            selection_probabilities
                            * selection_probabilities.clamp_min(1e-12).log()
                        ).sum(dim=0).mean()
                    else:
                        selected_timestep_index, selection_probabilities = (
                            select_teacher_guided_timesteps(
                                teacher_gaps=teacher_gap_matrix,
                                candidate_timesteps=candidate_timestep_tensor,
                                margin=float(config.get("tdsm_triplet_margin", 0.1)),
                                temperature=float(config.get("adaptive_timestep_temperature", 0.05)),
                                anchor_timestep=int(config.get("adaptive_timestep_anchor", 25)),
                                anchor_probability=float(
                                    config.get("adaptive_timestep_anchor_probability", 0.5)
                                ),
                            )
                        )
                    batch_indices = torch.arange(bsz, device=device)
                    timesteps = candidate_timestep_tensor[selected_timestep_index]
                    noisy = torch.stack(candidate_noisy)[selected_timestep_index, batch_indices]
                    target = torch.stack(candidate_target)[selected_timestep_index, batch_indices]
                    selected_teacher_gap = teacher_gap_matrix[
                        selected_timestep_index,
                        batch_indices,
                    ]
                    valid_count = random_valid.float().sum().clamp_min(1.0)
                    teacher_gap_mean = (
                        selected_teacher_gap * random_valid.float()
                    ).sum() / valid_count
                    teacher_valid_fraction = (
                        teacher_gap_matrix.gt(0.0)
                        .float()
                        .mul(random_valid.float().unsqueeze(0))
                        .sum()
                        / (valid_count * len(adaptive_timestep_candidates))
                    )
                    adaptive_timestep_mean = timesteps.float().mean()
                    selected_counts = torch.bincount(
                        selected_timestep_index,
                        minlength=len(adaptive_timestep_candidates),
                    ).detach().cpu().tolist()
                    for candidate_idx, count in enumerate(selected_counts):
                        epoch_timestep_counts[candidate_idx] += int(count)

                multipos_hidden = None
                neutral_align_hidden = None
                primitive_hidden_tokens = None
                primitive_uot_hidden_tokens = None
                clean_tide_states = None
                clean_tide_residuals = None
                if use_clean_tide_multilayer:
                    if clean_tide is None:
                        raise RuntimeError("D5b CleanTIDE student was not initialized")
                    clean_tide_states = clean_tide.encode(features)
                    clean_tide_residuals = clean_tide.condition_residuals(clean_tide_states)
                if use_neutral_align or use_primitive_multilabel:
                    neutral_fc = torch.zeros_like(fc)
                    detach_neutral_backbone = bool(
                        config.get(
                            "primitive_detach_backbone" if use_primitive_multilabel else "neutral_align_detach_backbone",
                            True,
                        )
                    )
                    # C10 stays detached by default.  Semi-UOT keeps this
                    # neutral DiT pass differentiable to supervise its tokens.
                    if detach_neutral_backbone and not use_primitive_semi_uot:
                        with torch.no_grad():
                            _, neutral_hidden_states = model(
                                noisy,
                                timesteps.to(dtype),
                                neutral_fc,
                                None,
                                return_hidden=True,
                                hidden_layers=[align_layer],
                                clean_token_residuals=clean_tide_residuals,
                            )
                    else:
                        _, neutral_hidden_states = model(
                            noisy,
                            timesteps.to(dtype),
                            neutral_fc,
                            None,
                            return_hidden=True,
                            hidden_layers=[align_layer],
                            clean_token_residuals=clean_tide_residuals,
                        )
                    if not neutral_hidden_states:
                        raise RuntimeError("Neutral auxiliary layer did not return a hidden state")
                    if use_neutral_align:
                        neutral_align_hidden = DiT.pool_hidden(neutral_hidden_states[0])
                        if detach_neutral_backbone:
                            neutral_align_hidden = neutral_align_hidden.detach()
                    if use_primitive_multilabel:
                        primitive_hidden_tokens = neutral_hidden_states[0]
                        if detach_neutral_backbone:
                            primitive_hidden_tokens = primitive_hidden_tokens.detach()
                        if use_primitive_semi_uot:
                            primitive_uot_hidden_tokens = neutral_hidden_states[0]
                if use_multipos_align:
                    pred, hidden_states = model(
                        noisy,
                        timesteps.to(dtype),
                        fc,
                        fl,
                        return_hidden=True,
                        hidden_layers=[align_layer],
                        clean_token_residuals=clean_tide_residuals,
                    )
                    if not hidden_states:
                        raise RuntimeError("C8 alignment layer did not return a hidden state")
                    multipos_hidden = DiT.pool_hidden(hidden_states[0])
                else:
                    pred = model(
                        noisy,
                        timesteps.to(dtype),
                        fc,
                        fl,
                        clean_token_residuals=clean_tide_residuals,
                    )
                if use_native_triplet:
                    pos_dist = reconstruction_distance(pred, target)
                else:
                    pred_sample = prediction_to_sample(
                        pred,
                        noisy,
                        timesteps,
                        scheduler,
                        prediction_type,
                    )
                    pos_dist = reconstruction_distance(pred_sample, features)

                pred_random = model(
                    noisy,
                    timesteps.to(dtype),
                    random_fc,
                    random_fl,
                    clean_token_residuals=clean_tide_residuals,
                )
                if use_native_triplet:
                    random_dist = reconstruction_distance(pred_random, target)
                else:
                    pred_random_sample = prediction_to_sample(
                        pred_random,
                        noisy,
                        timesteps,
                        scheduler,
                        prediction_type,
                    )
                    random_dist = reconstruction_distance(pred_random_sample, features)
                ambiguous_dist = None
                if use_ambiguity_contrast:
                    if ambiguous_neg_labels is None or ambiguous_valid is None:
                        raise RuntimeError("Ambiguity-aware negatives were not sampled")
                    ambiguous_fc, ambiguous_fl = split_text_condition(
                        text_embed,
                        ambiguous_neg_labels,
                    )
                    pred_ambiguous = model(
                        noisy,
                        timesteps.to(dtype),
                        ambiguous_fc,
                        ambiguous_fl,
                    )
                    pred_ambiguous_sample = prediction_to_sample(
                        pred_ambiguous,
                        noisy,
                        timesteps,
                        scheduler,
                        prediction_type,
                    )
                    ambiguous_dist = reconstruction_distance(pred_ambiguous_sample, features)
                second_random_dist = None
                if use_dual_random or use_ordinal_geometry:
                    if second_random_neg_labels is None or second_random_valid is None:
                        raise RuntimeError("Second random negative was not sampled")
                    second_random_fc, second_random_fl = split_text_condition(
                        text_embed,
                        second_random_neg_labels,
                    )
                    pred_second_random = model(
                        noisy,
                        timesteps.to(dtype),
                        second_random_fc,
                        second_random_fl,
                    )
                    pred_second_random_sample = prediction_to_sample(
                        pred_second_random,
                        noisy,
                        timesteps,
                        scheduler,
                        prediction_type,
                    )
                    second_random_dist = reconstruction_distance(
                        pred_second_random_sample,
                        features,
                    )
                is_hybrid = loss_mode in {"tdsm_x0_hybrid_triplet", "tdsm_x0_hybrid"}
                if is_hybrid:
                    if topk_hard_bank is not None:
                        hard_slot = min(
                            max(0, int(config.get("tdsm_hard_topk_slot", 0))),
                            int(topk_hard_bank.shape[1]) - 1,
                        )
                        hard_neg_labels = topk_hard_bank[labels, hard_slot]
                    elif hard_bank is not None:
                        hard_neg_labels = hard_bank[labels]
                    else:
                        hard_neg_labels = torch.roll(random_neg_labels, shifts=1)

                    hard_fc, hard_fl = split_text_condition(text_embed, hard_neg_labels)
                    pred_hard_neg = model(noisy, timesteps.to(dtype), hard_fc, hard_fl)
                    pred_hard_sample = prediction_to_sample(
                        pred_hard_neg, noisy, timesteps, scheduler, prediction_type
                    )
                    hard_dist = reconstruction_distance(pred_hard_sample, features)
                    hard_valid = labels.ne(hard_neg_labels)

                    inf = torch.full_like(random_dist, torch.inf)
                    random_dist = torch.where(random_valid, random_dist, inf)
                    hard_dist = torch.where(hard_valid, hard_dist, inf)
                    neg_dist = torch.minimum(random_dist, hard_dist)
                    mask = (random_valid | hard_valid).to(dtype=dtype)
                else:
                    neg_dist = random_dist
                    mask = random_valid.to(dtype=dtype)

                if use_timestep_gap_invariance:
                    gap_start_step = max(0, int(config.get("timestep_gap_start_step", 800)))
                    gap_warmup_steps = max(0, int(config.get("timestep_gap_warmup_steps", 800)))
                    if global_step >= gap_start_step:
                        partner_timestep = int(
                            random.choices(
                                timestep_gap_partners,
                                weights=timestep_gap_partner_weights,
                                k=1,
                            )[0]
                        )
                        gap_neg_labels = random_neg_labels
                        gap_valid_mask = random_valid
                        gap_fc, gap_fl = random_fc, random_fl
                        if use_reliable_timestep_gap:
                            # Hard seen-only negatives keep the consistency term
                            # focused on transferable decision boundaries rather
                            # than already-separated random pairs.
                            if topk_hard_bank is not None:
                                gap_slot = min(
                                    max(0, int(config.get("timestep_gap_hard_topk_slot", 0))),
                                    int(topk_hard_bank.shape[1]) - 1,
                                )
                                gap_neg_labels = topk_hard_bank[labels, gap_slot]
                            elif hard_bank is not None:
                                gap_neg_labels = hard_bank[labels]
                            gap_valid_mask = labels.ne(gap_neg_labels)
                            gap_fc, gap_fl = split_text_condition(text_embed, gap_neg_labels)
                        anchor_steps = torch.full(
                            (bsz,), timestep_gap_anchor, device=device, dtype=torch.long
                        )
                        partner_steps = torch.full(
                            (bsz,), partner_timestep, device=device, dtype=torch.long
                        )
                        # Use one epsilon trajectory for all time points.  The
                        # anchor is a stop-gradient self-teacher; only the
                        # partner branch changes the shared DiT parameters.
                        with torch.no_grad():
                            anchor_noisy = scheduler.add_noise(features, noise, anchor_steps)
                            anchor_positive = model(
                                anchor_noisy, anchor_steps.to(dtype), fc, fl
                            )
                            anchor_negative = model(
                                anchor_noisy, anchor_steps.to(dtype), gap_fc, gap_fl
                            )
                            anchor_gap = (
                                reconstruction_distance(
                                    prediction_to_sample(
                                        anchor_negative,
                                        anchor_noisy,
                                        anchor_steps,
                                        scheduler,
                                        prediction_type,
                                    ),
                                    features,
                                )
                                - reconstruction_distance(
                                    prediction_to_sample(
                                        anchor_positive,
                                        anchor_noisy,
                                        anchor_steps,
                                        scheduler,
                                        prediction_type,
                                    ),
                                    features,
                                )
                            )
                        partner_noisy = scheduler.add_noise(features, noise, partner_steps)
                        partner_positive = model(
                            partner_noisy, partner_steps.to(dtype), fc, fl
                        )
                        partner_negative = model(
                            partner_noisy, partner_steps.to(dtype), gap_fc, gap_fl
                        )
                        partner_gap = (
                            reconstruction_distance(
                                prediction_to_sample(
                                    partner_negative,
                                    partner_noisy,
                                    partner_steps,
                                    scheduler,
                                    prediction_type,
                                ),
                                features,
                            )
                            - reconstruction_distance(
                                prediction_to_sample(
                                    partner_positive,
                                    partner_noisy,
                                    partner_steps,
                                    scheduler,
                                    prediction_type,
                                ),
                                features,
                            )
                        )
                        (
                            timestep_gap_loss,
                            timestep_gap_coverage,
                            timestep_gap_anchor_mean,
                            timestep_gap_partner_mean,
                        ) = scale_calibrated_timestep_gap_loss(
                            anchor_gap=anchor_gap,
                            partner_gap=partner_gap,
                            valid_mask=gap_valid_mask,
                            anchor_margin=float(config.get("timestep_gap_reliable_margin", 0.10)),
                            partner_floor=float(config.get("timestep_gap_partner_floor", 0.0)),
                            huber_delta=float(config.get("timestep_gap_huber_delta", 0.10)),
                            max_anchor_gap=timestep_gap_hard_max,
                        )
                        warmup_progress = (
                            1.0
                            if gap_warmup_steps == 0
                            else min(
                                1.0,
                                float(global_step - gap_start_step + 1) / gap_warmup_steps,
                            )
                        )
                        timestep_gap_weight = timestep_gap_base_weight * warmup_progress

                noise_consistent_recon = None
                positive_distance_views = [pos_dist]
                negative_distance_views = [neg_dist]
                positive_recon_views = [F.mse_loss(pred, target)]
                if use_noise_consistent:
                    for _view_idx in range(1, noise_consistent_views):
                        view_noise = torch.randn_like(features)
                        view_noisy = scheduler.add_noise(features, view_noise, timesteps)
                        view_target = diffusion_training_target(
                            prediction_type,
                            features,
                            view_noise,
                            timesteps,
                            scheduler,
                        )
                        view_pred = model(view_noisy, timesteps.to(dtype), fc, fl)
                        view_negative_pred = model(
                            view_noisy,
                            timesteps.to(dtype),
                            random_fc,
                            random_fl,
                        )
                        view_pred_sample = prediction_to_sample(
                            view_pred,
                            view_noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        view_negative_sample = prediction_to_sample(
                            view_negative_pred,
                            view_noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        positive_distance_views.append(
                            reconstruction_distance(view_pred_sample, features)
                        )
                        negative_distance_views.append(
                            reconstruction_distance(view_negative_sample, features)
                        )
                        positive_recon_views.append(F.mse_loss(view_pred, view_target))
                    noise_consistent_recon = torch.stack(positive_recon_views).mean()

                agreement_view_gap = None
                if use_agreement_weighted_triplet:
                    # This detached second view selects stable semi-hard pairs;
                    # it does not impose C3-style cross-view energy matching.
                    with torch.no_grad():
                        agreement_noise = torch.randn_like(features)
                        agreement_noisy = scheduler.add_noise(
                            features,
                            agreement_noise,
                            timesteps,
                        )
                        agreement_positive = model(
                            agreement_noisy,
                            timesteps.to(dtype),
                            fc,
                            fl,
                        )
                        agreement_negative = model(
                            agreement_noisy,
                            timesteps.to(dtype),
                            random_fc,
                            random_fl,
                        )
                        agreement_positive_sample = prediction_to_sample(
                            agreement_positive,
                            agreement_noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        agreement_negative_sample = prediction_to_sample(
                            agreement_negative,
                            agreement_noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        agreement_view_gap = (
                            reconstruction_distance(agreement_negative_sample, features)
                            - reconstruction_distance(agreement_positive_sample, features)
                        )

                base_margin = float(
                    config.get(
                        "tdsm_triplet_margin",
                        config.get("margin_x0", config.get("margin", 0.1)),
                    )
                )
                margin: float | torch.Tensor = base_margin
                if use_natural_ambiguity:
                    if sampled_ambiguity_mask is None:
                        raise RuntimeError("Natural ambiguity mask was not sampled")
                    margin = torch.full_like(pos_dist, base_margin)
                    margin = torch.where(
                        sampled_ambiguity_mask,
                        torch.full_like(margin, float(config.get("ambiguity_margin", 0.05))),
                        margin,
                    )
                if use_noise_consistent:
                    triplet_x0, noise_gap_mean, noise_gap_std, rank_active_fraction = (
                        noise_consistent_energy_ranking_loss(
                            positive_distances=torch.stack(positive_distance_views),
                            negative_distances=torch.stack(negative_distance_views),
                            valid_mask=random_valid,
                            margin=base_margin,
                            beta=noise_consistent_beta,
                        )
                    )
                elif use_adaptive_margin:
                    if prototype_direction_context is None:
                        raise RuntimeError("Adaptive margin context was not initialized")
                    margin, adaptive_margin_mean, adaptive_margin_reliability = (
                        reliability_gated_adaptive_margin(
                            labels=labels,
                            negative_labels=random_neg_labels,
                            context=prototype_direction_context,
                            base_margin=base_margin,
                            min_margin=float(config.get("adaptive_margin_min", 0.05)),
                            max_margin=float(config.get("adaptive_margin_max", 0.15)),
                        )
                    )
                    margin = margin.to(dtype=pos_dist.dtype)
                if not use_noise_consistent:
                    rank_violation = pos_dist - neg_dist + margin
                    triplet_x0 = torch.clamp(rank_violation, min=0.0) * mask
                    rank_active_fraction = (
                        (rank_violation.gt(0).to(dtype=mask.dtype) * mask).sum()
                        / mask.sum().clamp_min(1.0)
                    )
                    if use_dual_random:
                        if second_random_dist is None or second_random_valid is None:
                            raise RuntimeError("Second random distance was not computed")
                        second_mask = second_random_valid.to(dtype=pos_dist.dtype)
                        second_violation = pos_dist - second_random_dist + base_margin
                        second_triplet = torch.clamp(second_violation, min=0.0) * second_mask
                        second_active = (
                            second_violation.gt(0).to(dtype=second_mask.dtype) * second_mask
                        ).sum() / second_mask.sum().clamp_min(1.0)
                        triplet_x0 = 0.5 * (triplet_x0 + second_triplet)
                        rank_active_fraction = 0.5 * (
                            rank_active_fraction + second_active
                        )
                    if use_ambiguity_contrast:
                        if ambiguous_dist is None or ambiguous_valid is None:
                            raise RuntimeError("Ambiguity-aware distances were not computed")
                        ambiguous_mask = ambiguous_valid.to(dtype=pos_dist.dtype)
                        ambiguity_coverage = ambiguous_mask.mean()
                        ambiguous_violation = (
                            pos_dist
                            - ambiguous_dist
                            + float(config.get("ambiguity_margin", 0.0))
                        )
                        ambiguity_rank_loss = (
                            torch.clamp(ambiguous_violation, min=0.0) * ambiguous_mask
                        ).sum() / ambiguous_mask.sum().clamp_min(1.0)
                        ambiguity_active_fraction = (
                            ambiguous_violation.gt(0).to(dtype=ambiguous_mask.dtype)
                            * ambiguous_mask
                        ).sum() / ambiguous_mask.sum().clamp_min(1.0)
                    elif use_natural_ambiguity:
                        if sampled_ambiguity_mask is None:
                            raise RuntimeError("Natural ambiguity mask was not sampled")
                        natural_mask = (
                            sampled_ambiguity_mask.to(dtype=pos_dist.dtype)
                            * random_valid.to(dtype=pos_dist.dtype)
                        )
                        ambiguity_coverage = natural_mask.sum() / random_valid.to(
                            dtype=pos_dist.dtype
                        ).sum().clamp_min(1.0)
                        ambiguity_rank_loss = (
                            torch.clamp(rank_violation, min=0.0) * natural_mask
                        ).sum() / natural_mask.sum().clamp_min(1.0)
                        ambiguity_active_fraction = (
                            rank_violation.gt(0).to(dtype=natural_mask.dtype) * natural_mask
                        ).sum() / natural_mask.sum().clamp_min(1.0)

                agreement_triplet = torch.tensor(0.0, device=device)
                if use_agreement_weighted_triplet:
                    if agreement_view_gap is None:
                        raise RuntimeError("Agreement-weighted triplet view was not computed")
                    primary_gap = neg_dist - pos_dist
                    stable_pair = (
                        primary_gap.detach().gt(0.0)
                        & agreement_view_gap.gt(0.0)
                        & primary_gap.detach().sub(agreement_view_gap).abs().le(
                            agreement_gap_tolerance
                        )
                    ).to(dtype=pos_dist.dtype) * mask
                    rank_agreement_coverage = stable_pair.sum() / mask.sum().clamp_min(1.0)
                    agreement_triplet = torch.clamp(
                        margin - primary_gap,
                        min=0.0,
                    ).mul(stable_pair).mean()

                semantic_weight = 0.0
                if loss_mode in semantic_distill_modes:
                    if semantic_neighbor_bank is None or semantic_text_features is None:
                        raise RuntimeError("Semantic topology context was not initialized")
                    semantic_labels = semantic_neighbor_bank[labels]
                    if bool((semantic_labels < 0).any()):
                        raise RuntimeError("Semantic neighbour bank is missing a seen training label")

                    topology_labels = [labels]
                    topology_distances = [pos_dist]
                    if int(config.get("semantic_random_neg", 1)) == 1:
                        topology_labels.append(random_neg_labels)
                        topology_distances.append(random_dist)
                    for neighbor_col in range(semantic_labels.shape[1]):
                        neighbor_labels = semantic_labels[:, neighbor_col]
                        neighbor_fc, neighbor_fl = split_text_condition(text_embed, neighbor_labels)
                        pred_neighbor = model(noisy, timesteps.to(dtype), neighbor_fc, neighbor_fl)
                        pred_neighbor_sample = prediction_to_sample(
                            pred_neighbor,
                            noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        topology_labels.append(neighbor_labels)
                        topology_distances.append(
                            reconstruction_distance(pred_neighbor_sample, features)
                        )

                    semantic_kd = semantic_topology_distillation_loss(
                        distance_matrix=torch.stack(topology_distances, dim=1),
                        candidate_labels=torch.stack(topology_labels, dim=1),
                        anchor_labels=labels,
                        normalized_text=semantic_text_features,
                        teacher_tau=float(config.get("semantic_teacher_tau", 0.1)),
                        student_tau=float(config.get("semantic_student_tau", 0.05)),
                    )
                    semantic_weight = semantic_distill_curriculum_weight(
                        step=global_step + 1,
                        peak_weight=float(config.get("semantic_distill_weight", 0.1)),
                        warmup_steps=int(config.get("semantic_distill_warmup_steps", 1000)),
                        decay_end_steps=int(
                            config.get("semantic_distill_decay_end_steps", 0)
                        ),
                    )

                prototype_direction_weight = 0.0
                if use_prototype_direction:
                    if prototype_direction_context is None:
                        raise RuntimeError("Prototype direction context was not initialized")
                    prototype_direction, prototype_reliability = prototype_direction_contrastive_loss(
                        positive_sample=pred_sample,
                        negative_sample=pred_random_sample,
                        labels=labels,
                        negative_labels=random_neg_labels,
                        context=prototype_direction_context,
                    )
                    direction_warmup = max(
                        0,
                        int(config.get("prototype_direction_warmup_steps", 1000)),
                    )
                    warmup_scale = (
                        min(1.0, float(global_step + 1) / float(direction_warmup))
                        if direction_warmup > 0
                        else 1.0
                    )
                    prototype_direction_weight = float(
                        config.get("prototype_direction_weight", 0.01)
                    ) * warmup_scale

                loss_aware_main_loss = None
                if use_loss_aware_timestep or use_adaptive_evidence_timestep:
                    if noise_consistent_recon is not None:
                        raise RuntimeError("Loss-aware sampling cannot be combined with noise-consistent reconstruction")
                    recon_per_sample = F.mse_loss(
                        pred,
                        target,
                        reduction="none",
                    ).flatten(1).mean(dim=1)
                    recon = recon_per_sample.mean()
                elif use_band_elbo_snr:
                    if noise_consistent_recon is not None:
                        raise RuntimeError("Band-tempered ELBO cannot be combined with noise-consistent reconstruction")
                    recon = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1).mean()
                else:
                    recon = (
                        noise_consistent_recon
                        if noise_consistent_recon is not None
                        else F.mse_loss(pred, target)
                    )
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = (
                    (triplet_x0 * elbo_uniform_importance).mean()
                    if use_band_elbo_snr
                    else (
                        (triplet_x0 * loss_aware_importance).mean()
                        if use_adaptive_evidence_timestep
                        else triplet_x0.mean()
                    )
                )
                if use_agreement_weighted_triplet:
                    hard_rank = hard_rank + agreement_triplet * agreement_triplet_weight
                if use_ambiguity_contrast:
                    hard_rank = hard_rank + ambiguity_rank_loss * float(
                        config.get("ambiguity_rank_weight", 0.25)
                    )
                if use_loss_aware_timestep or use_adaptive_evidence_timestep:
                    # This is the uniform-timestep main objective estimated
                    # under p(t), not a timestep reweighting objective.
                    loss_aware_per_sample = (
                        recon_per_sample * float(config.get("rec_weight", config.get("d_weight", 1.0)))
                        + triplet_x0 * float(
                            config.get("tdsm_triplet_weight", config.get("t_weight", 1.0))
                        )
                    )
                    if use_loss_aware_timestep:
                        with torch.no_grad():
                            for timestep_value in timesteps.unique().tolist():
                                timestep_index = int(timestep_value)
                                timestep_mask = timesteps.eq(timestep_index)
                                observed_second_moment = loss_aware_per_sample[
                                    timestep_mask
                                ].detach().square().mean().float()
                                if int(loss_aware_observation_count[timestep_index].item()) == 0:
                                    loss_aware_second_moment[timestep_index] = observed_second_moment
                                else:
                                    loss_aware_second_moment[timestep_index].mul_(
                                        loss_aware_ema_decay
                                    ).add_(
                                        observed_second_moment,
                                        alpha=1.0 - loss_aware_ema_decay,
                                    )
                                loss_aware_observation_count[timestep_index].add_(
                                    int(timestep_mask.sum().item())
                                )
                    loss_aware_main_loss = (
                        loss_aware_per_sample * loss_aware_importance
                    ).mean()
                infonce = torch.tensor(0.0, device=device)
                projected_infonce = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                multipos_weight = 0.0
                neutral_align_weight = 0.0
                primitive_multilabel_weight = 0.0
                primitive_proto_contrast_weight = 0.0
                primitive_tmr_weight = 0.0
                primitive_feature_nll_weight = 0.0
                primitive_feature_recon_weight = 0.0
                clean_tide_primitive_weight = 0.0
                clean_tide_distill_weight = 0.0
                episodic_proto_mse_weight = 0.0
                episodic_proto_nll_weight = 0.0
                counterfactual_text_weight = 0.0
                energy_calibration_weight = 0.0
                primitive_uot_weight = 0.0
                ordinal_geometry = torch.tensor(0.0, device=device)
                ordinal_geometry_weight = 0.0
                ordinal_geometry_coverage = torch.tensor(0.0, device=device)
                if use_multipos_align:
                    if aux is None or multipos_hidden is None:
                        raise RuntimeError("C8 multi-positive alignment context was not initialized")
                    text_align, multipos_snr_mean = multipositive_bidirectional_alignment_loss(
                        aux=aux,
                        hidden=multipos_hidden,
                        text_embed=text_embed,
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        timesteps=timesteps,
                        scheduler=scheduler,
                        tau=float(config.get("multipos_align_tau", 0.07)),
                        snr_floor=float(config.get("multipos_snr_floor", 0.05)),
                    )
                    warmup_steps = max(
                        0,
                        int(config.get("multipos_align_warmup_steps", 1000)),
                    )
                    warmup_scale = (
                        min(1.0, float(global_step + 1) / float(warmup_steps))
                        if warmup_steps > 0
                        else 1.0
                    )
                    multipos_weight = float(config.get("multipos_align_weight", 0.05)) * warmup_scale
                if use_neutral_align:
                    if aux is None or neutral_align_hidden is None:
                        raise RuntimeError("C9 neutral alignment context was not initialized")
                    text_align, neutral_negative_weight = neutral_debiased_alignment_loss(
                        aux=aux,
                        hidden=neutral_align_hidden,
                        text_embed=text_embed,
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        tau=float(config.get("neutral_align_tau", 0.07)),
                        negative_floor=float(config.get("neutral_align_negative_floor", 0.1)),
                        negative_power=float(config.get("neutral_align_negative_power", 1.0)),
                    )
                    neutral_warmup_steps = max(
                        0,
                        int(config.get("neutral_align_warmup_steps", 1000)),
                    )
                    neutral_warmup_scale = (
                        min(1.0, float(global_step + 1) / float(neutral_warmup_steps))
                        if neutral_warmup_steps > 0
                        else 1.0
                    )
                    neutral_align_weight = float(
                        config.get("neutral_align_weight", 0.05)
                    ) * neutral_warmup_scale
                if use_primitive_multilabel:
                    if aux is None or primitive_hidden_tokens is None or primitive_context is None:
                        raise RuntimeError("C10 primitive multi-label context was not initialized")
                    primitive_logits = action_primitive_logits(
                        aux,
                        primitive_hidden_tokens,
                        primitive_context,
                        logit_scale=float(config.get("primitive_logit_scale", 10.0)),
                        logit_bias=float(config.get("primitive_logit_bias", -2.0)),
                    )
                    if use_primitive_nnpu:
                        (
                            text_align,
                            primitive_target_density,
                            primitive_pu_active,
                            primitive_pu_corrected_negative,
                        ) = action_primitive_nnpu_loss(
                            primitive_logits,
                            labels,
                            primitive_context,
                            prior_floor=float(config.get("primitive_pu_prior_floor", 0.01)),
                            prior_ceiling=float(config.get("primitive_pu_prior_ceiling", 0.99)),
                        )
                    else:
                        text_align, primitive_target_density = action_primitive_multilabel_loss(
                            primitive_logits,
                            labels,
                            primitive_context,
                            negative_gamma=float(config.get("primitive_negative_gamma", 2.0)),
                            negative_weight=float(config.get("primitive_negative_loss_weight", 0.25)),
                            sample_weight=(
                                elbo_uniform_importance
                                if use_band_elbo_snr
                                else loss_aware_importance
                                if (use_loss_aware_timestep or use_adaptive_evidence_timestep)
                                else None
                            ),
                        )
                    primitive_warmup_steps = max(
                        0,
                        int(config.get("primitive_warmup_steps", 1000)),
                    )
                    primitive_warmup_scale = (
                        min(1.0, float(global_step + 1) / float(primitive_warmup_steps))
                        if primitive_warmup_steps > 0
                        else 1.0
                    )
                    primitive_multilabel_weight = float(
                        config.get("primitive_multilabel_weight", 0.05)
                    ) * primitive_warmup_scale

                    if use_primitive_semi_uot:
                        if primitive_uot_hidden_tokens is None:
                            raise RuntimeError("Semi-UOT did not receive differentiable DiT hidden tokens")
                        primitive_uot_logits = action_primitive_token_logits(
                            aux,
                            primitive_uot_hidden_tokens,
                            primitive_context,
                            logit_scale=float(config.get("primitive_logit_scale", 10.0)),
                            logit_bias=float(config.get("primitive_logit_bias", -2.0)),
                        )
                        primitive_uot, primitive_uot_target_l1 = semi_uot_primitive_coverage_loss(
                            primitive_uot_logits,
                            labels,
                            primitive_context,
                            epsilon=float(config.get("primitive_uot_epsilon", 0.10)),
                            target_tau=float(config.get("primitive_uot_target_tau", 0.20)),
                            iterations=int(config.get("primitive_uot_iterations", 20)),
                        )
                        uot_warmup_steps = max(0, int(config.get("primitive_uot_warmup_steps", 1000)))
                        uot_warmup_scale = (
                            min(1.0, float(global_step + 1) / float(uot_warmup_steps))
                            if uot_warmup_steps > 0
                            else 1.0
                        )
                        primitive_uot_weight = float(
                            config.get("primitive_uot_weight", 0.01)
                        ) * uot_warmup_scale

                    if use_primitive_proto_contrast:
                        (
                            primitive_proto_contrast,
                            primitive_proto_negative_weight,
                        ) = action_primitive_debiased_prototype_contrast_loss(
                            primitive_logits,
                            labels,
                            primitive_context,
                            tau=float(config.get("primitive_proto_contrast_tau", 0.1)),
                            negative_floor=float(
                                config.get("primitive_proto_negative_floor", 0.05)
                            ),
                            negative_power=float(
                                config.get("primitive_proto_negative_power", 1.0)
                            ),
                        )
                        primitive_proto_contrast_weight = float(
                            config.get("primitive_proto_contrast_weight", 0.02)
                        ) * primitive_warmup_scale
                    if use_primitive_tmr_filtered:
                        primitive_tmr_infonce, primitive_tmr_filtered = (
                            tmr_filtered_primitive_text_infonce_loss(
                                aux=aux,
                                hidden=DiT.pool_hidden(primitive_hidden_tokens),
                                text_embed=text_embed,
                                labels=labels,
                                seen_labels=seen_tensor_choices,
                                tau=float(config.get("primitive_tmr_tau", 0.1)),
                                similarity_threshold=float(
                                    config.get("primitive_tmr_text_similarity_threshold", 0.8)
                                ),
                            )
                        )
                        primitive_tmr_weight = float(
                            config.get("primitive_tmr_weight", 0.02)
                        ) * primitive_warmup_scale
                    if use_primitive_feature_distribution:
                        (
                            primitive_feature_nll,
                            primitive_feature_recon,
                            primitive_feature_variance,
                        ) = primitive_feature_distribution_loss(
                            aux=aux,
                            features=features,
                            labels=labels,
                            primitive_context=primitive_context,
                        )
                        primitive_feature_nll_weight = float(
                            config.get("primitive_feature_distribution_nll_weight", 0.03)
                        ) * primitive_warmup_scale
                        primitive_feature_recon_weight = float(
                            config.get("primitive_feature_distribution_recon_weight", 0.02)
                        ) * primitive_warmup_scale
                    if use_episodic_primitive_proto:
                        if episodic_proto_context is None:
                            raise RuntimeError("C19 visual prototype context was not initialized")
                        episodic_proto_mse, episodic_proto_nll = episodic_prototype_completion_loss(
                            aux=aux,
                            labels=labels,
                            primitive_context=primitive_context,
                            visual_context=episodic_proto_context,
                            temperature=float(config.get("episodic_proto_temperature", 0.2)),
                        )
                        episodic_proto_mse_weight = float(
                            config.get("episodic_proto_mse_weight", 0.05)
                        ) * primitive_warmup_scale
                        episodic_proto_nll_weight = float(
                            config.get("episodic_proto_nll_weight", 0.01)
                        ) * primitive_warmup_scale
                    if use_counterfactual_text:
                        counterfactual_fc, counterfactual_fl = counterfactual_text_condition(
                            text_embed,
                            labels,
                            primitive_context,
                        )
                        counterfactual_pred = model(
                            noisy,
                            timesteps.to(dtype),
                            counterfactual_fc,
                            counterfactual_fl,
                        )
                        counterfactual_sample = prediction_to_sample(
                            counterfactual_pred,
                            noisy,
                            timesteps,
                            scheduler,
                            prediction_type,
                        )
                        counterfactual_distance = reconstruction_distance(
                            counterfactual_sample,
                            features,
                        )
                        counterfactual_text_gap = (counterfactual_distance - pos_dist).mean()
                        counterfactual_text_loss = F.relu(
                            pos_dist - counterfactual_distance
                            + float(config.get("counterfactual_text_margin", 0.02))
                        ).mean()
                        counterfactual_text_weight = float(
                            config.get("counterfactual_text_weight", 0.05)
                        ) * primitive_warmup_scale
                    if use_clean_tide:
                        if clean_tide is None or clean_tide_teacher is None:
                            raise RuntimeError("D5 CleanTIDE student or frozen teacher was not initialized")
                        clean_tokens = clean_tide.encode(features)
                        clean_tide_logits = action_primitive_logits(
                            aux,
                            clean_tokens,
                            primitive_context,
                            logit_scale=float(config.get("primitive_logit_scale", 10.0)),
                            logit_bias=float(config.get("primitive_logit_bias", -2.0)),
                        )
                        clean_tide_primitive, _ = action_primitive_multilabel_loss(
                            clean_tide_logits,
                            labels,
                            primitive_context,
                            negative_gamma=float(config.get("primitive_negative_gamma", 2.0)),
                            negative_weight=float(config.get("primitive_negative_loss_weight", 0.25)),
                        )
                        stratum_losses: list[torch.Tensor] = []
                        stratum_cosines: list[torch.Tensor] = []
                        for stratum_index, (low_t, high_t) in enumerate(clean_tide_strata):
                            teacher_timesteps = torch.randint(
                                int(low_t), int(high_t) + 1, (bsz,), device=device
                            ).long()
                            teacher_noise = torch.randn_like(features)
                            teacher_noisy = scheduler.add_noise(
                                features, teacher_noise, teacher_timesteps
                            )
                            teacher_neutral_fc = torch.zeros(
                                (bsz, int(text_embed.shape[-1])), device=device, dtype=dtype
                            )
                            with torch.no_grad():
                                _, teacher_hidden_states = clean_tide_teacher(
                                    teacher_noisy,
                                    teacher_timesteps.to(dtype),
                                    teacher_neutral_fc,
                                    None,
                                    return_hidden=True,
                                    hidden_layers=[align_layer],
                                )
                            if not teacher_hidden_states:
                                raise RuntimeError("D5 teacher did not return the requested alignment layer")
                            teacher_tokens = teacher_hidden_states[0].detach()
                            student_tokens = clean_tide.project(clean_tokens, stratum_index)
                            cosine = F.cosine_similarity(
                                student_tokens.float(), teacher_tokens.float(), dim=-1
                            ).mean()
                            stratum_cosines.append(cosine)
                            stratum_losses.append(1.0 - cosine)
                        clean_tide_distill = torch.stack(stratum_losses).mean()
                        clean_tide_cosine = torch.stack(stratum_cosines).mean()
                        clean_warmup_steps = max(
                            0, int(config.get("clean_tide_warmup_steps", 1000))
                        )
                        clean_warmup = (
                            min(1.0, float(global_step + 1) / float(clean_warmup_steps))
                            if clean_warmup_steps > 0
                            else 1.0
                        )
                        clean_tide_primitive_weight = float(
                            config.get("clean_tide_primitive_weight", 0.05)
                        ) * clean_warmup
                        clean_tide_distill_weight = float(
                            config.get("clean_tide_distill_weight", 0.05)
                        ) * clean_warmup
                    if use_clean_tide_multilayer:
                        if (
                            clean_tide is None
                            or clean_tide_teacher is None
                            or clean_tide_states is None
                            or clean_tide_residuals is None
                        ):
                            raise RuntimeError("D5b CleanTIDE context was not initialized")
                        stratum_losses: list[torch.Tensor] = []
                        stratum_cosines: list[torch.Tensor] = []
                        for stratum_index, bucket in enumerate(clean_tide_buckets):
                            generator = torch.Generator(device=device).manual_seed(
                                int(config.get("seed", 2026)) * 1000003
                                + (global_step + 1) * 97
                                + stratum_index
                            )
                            bucket_positions = torch.randint(
                                0,
                                int(bucket.numel()),
                                (bsz,),
                                device=device,
                                generator=generator,
                            )
                            teacher_timesteps = bucket[bucket_positions]
                            teacher_noise = torch.randn(
                                features.shape,
                                device=device,
                                dtype=dtype,
                                generator=generator,
                            )
                            teacher_neutral_fc = torch.zeros(
                                (bsz, int(text_embed.shape[-1])), device=device, dtype=dtype
                            )
                            with torch.no_grad():
                                _, teacher_positive_states = clean_tide_teacher(
                                    scheduler.add_noise(features, teacher_noise, teacher_timesteps),
                                    teacher_timesteps.to(dtype),
                                    teacher_neutral_fc,
                                    None,
                                    return_hidden=True,
                                    hidden_layers=clean_tide_layers,
                                )
                                _, teacher_negative_states = clean_tide_teacher(
                                    scheduler.add_noise(features, -teacher_noise, teacher_timesteps),
                                    teacher_timesteps.to(dtype),
                                    teacher_neutral_fc,
                                    None,
                                    return_hidden=True,
                                    hidden_layers=clean_tide_layers,
                                )
                            for layer, teacher_positive, teacher_negative in zip(
                                clean_tide_layers,
                                teacher_positive_states,
                                teacher_negative_states,
                            ):
                                student_tokens = clean_tide.project(
                                    clean_tide_states[layer], layer, stratum_index
                                )
                                teacher_tokens = 0.5 * (teacher_positive.detach() + teacher_negative.detach())
                                cosine = F.cosine_similarity(
                                    student_tokens.float(), teacher_tokens.float(), dim=-1
                                ).mean()
                                stratum_cosines.append(cosine)
                                stratum_losses.append(1.0 - cosine)
                        clean_tide_distill = torch.stack(stratum_losses).mean()
                        clean_tide_cosine = torch.stack(stratum_cosines).mean()
                        clean_warmup_steps = max(
                            0, int(config.get("clean_tide_warmup_steps", 2000))
                        )
                        clean_warmup = (
                            min(1.0, float(global_step + 1) / float(clean_warmup_steps))
                            if clean_warmup_steps > 0
                            else 1.0
                        )
                        clean_tide_distill_weight = float(
                            config.get("clean_tide_distill_weight", 0.05)
                        ) * clean_warmup
                if use_frozen_gallery_match:
                    if aux is None:
                        raise RuntimeError("F1 frozen-gallery matcher was not initialized")
                    f1_gallery_match, f1_csv_gate = frozen_gallery_match_loss(
                        aux=aux,
                        features=features,
                        text_embed=text_embed,
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        temperature=float(config.get("f1_match_temperature", 0.10)),
                    )
                    f1_warmup_steps = max(0, int(config.get("f1_match_warmup_steps", 1000)))
                    f1_warmup = (
                        min(1.0, float(global_step + 1) / float(f1_warmup_steps))
                        if f1_warmup_steps > 0
                        else 1.0
                    )
                    f1_gallery_match_weight = float(
                        config.get("f1_match_weight", 0.10)
                    ) * f1_warmup
                if use_energy_calibration:
                    # Seen labels are observed, so this is a proper conditional
                    # energy likelihood, not contrastive learning on pseudo
                    # unseen labels.  The DiT's own reconstruction energy is
                    # used directly as the class logit.
                    energy_negative_labels = sample_infonce_negative_labels(
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        num_neg=max(1, int(config.get("energy_calibration_num_neg", 2))),
                        device=device,
                    )
                    energy_calibration, _ = reconstruction_infonce_loss(
                        model=model,
                        noisy=noisy,
                        timesteps=timesteps.to(dtype),
                        text_embed=text_embed,
                        labels=labels,
                        negative_labels=energy_negative_labels,
                        target=features,
                        scheduler=scheduler,
                        prediction_type=prediction_type,
                        tau=max(float(config.get("energy_calibration_tau", 0.05)), 1e-6),
                    )
                    calibration_warmup_steps = max(
                        0,
                        int(config.get("energy_calibration_warmup_steps", 2000)),
                    )
                    calibration_warmup = (
                        min(1.0, float(global_step + 1) / float(calibration_warmup_steps))
                        if calibration_warmup_steps > 0
                        else 1.0
                    )
                    energy_calibration_weight = float(
                        config.get("energy_calibration_weight", 0.05)
                    ) * calibration_warmup
                if use_multitime_energy_distill:
                    if multitime_energy_teacher is None:
                        raise RuntimeError("Multi-timestep energy teacher was not initialized")
                    distill_negative_labels = sample_infonce_negative_labels(
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        num_neg=max(1, int(config.get("energy_distill_num_neg", 2))),
                        device=device,
                        hard_bank=hard_bank,
                        topk_hard_bank=topk_hard_bank,
                        force_hard_pairs=force_hard_pairs,
                    )
                    distill_candidate_labels = torch.cat(
                        (labels.unsqueeze(1), distill_negative_labels), dim=1
                    )
                    (
                        energy_distill_kl,
                        energy_distill_gap,
                        energy_distill_agreement,
                    ) = multitime_energy_distillation_loss(
                        student=model,
                        teacher=multitime_energy_teacher,
                        features=features,
                        text_embed=text_embed,
                        candidate_labels=distill_candidate_labels,
                        scheduler=scheduler,
                        prediction_type=prediction_type,
                        anchor_timestep=energy_distill_anchor_timestep,
                        teacher_timesteps=energy_distill_teacher_timesteps,
                        teacher_weights=energy_distill_teacher_weights,
                        teacher_temperature=float(config.get("energy_distill_teacher_temperature", 0.25)),
                        student_temperature=float(config.get("energy_distill_student_temperature", 0.25)),
                        agreement_gate=bool(config.get("energy_distill_agreement_gate", True)),
                        gap_delta=float(config.get("energy_distill_gap_delta", 0.25)),
                    )
                    distill_warmup_steps = max(
                        0, int(config.get("energy_distill_warmup_steps", 1000))
                    )
                    distill_warmup = (
                        min(1.0, float(global_step + 1) / float(distill_warmup_steps))
                        if distill_warmup_steps > 0
                        else 1.0
                    )
                    energy_distill_weight = float(
                        config.get("energy_distill_kl_weight", 0.05)
                    ) * distill_warmup
                    energy_distill_gap_weight = float(
                        config.get("energy_distill_gap_weight", 0.02)
                    ) * distill_warmup
                if use_ordinal_geometry:
                    if (
                        c23_geometry_distance is None
                        or second_random_neg_labels is None
                        or second_random_valid is None
                        or second_random_dist is None
                    ):
                        raise RuntimeError("C23 ordinal geometry context was not initialized")
                    first_geometry = c23_geometry_distance[labels, random_neg_labels]
                    second_geometry = c23_geometry_distance[labels, second_random_neg_labels]
                    geometry_delta = second_geometry - first_geometry
                    ordinal_valid = (
                        random_valid
                        & second_random_valid
                        & geometry_delta.abs().ge(
                            float(config.get("c23_min_geometry_gap", 0.05))
                        )
                    )
                    energy_order_all = second_random_dist.detach() - random_dist.detach()
                    if use_reliable_ordinal_geometry:
                        ordinal_valid = ordinal_valid & energy_order_all.abs().le(
                            float(config.get("c23_max_energy_gap", 0.05))
                        )
                    reliability_target = None
                    if use_soft_ordinal_geometry:
                        if c24_reliability is None:
                            raise RuntimeError("C24 reliability context was not initialized")
                        reliability_target = c24_reliability[labels]
                        ordinal_valid = ordinal_valid & reliability_target.ge(
                            float(config.get("c24_min_reliability", 0.55))
                        )
                    ordinal_geometry_coverage = ordinal_valid.float().mean()
                    if ordinal_valid.any():
                        energy_order = (
                            second_random_dist[ordinal_valid] - random_dist[ordinal_valid]
                        )
                        order_logit = geometry_delta[ordinal_valid].sign() * energy_order / max(
                            float(config.get("c23_ordinal_temperature", 0.10)), 1e-6
                        )
                        if use_soft_ordinal_geometry:
                            ordinal_geometry = F.binary_cross_entropy_with_logits(
                                order_logit,
                                reliability_target[ordinal_valid],
                            )
                        else:
                            ordinal_geometry = F.softplus(-order_logit).mean()
                    ordinal_warmup = max(0, int(config.get("c23_warmup_steps", 1000)))
                    ordinal_geometry_weight = float(
                        config.get("c23_ordinal_weight", 0.02)
                    ) * (
                        min(1.0, float(global_step + 1) / float(ordinal_warmup))
                        if ordinal_warmup > 0
                        else 1.0
                    )
                if use_episodic_gallery and global_step >= int(
                    config.get("episodic_gallery_start_step", 1000)
                ):
                    interval = max(1, int(config.get("episodic_gallery_interval", 4)))
                    if global_step % interval == 0:
                        (
                            episodic_gallery_rank,
                            episodic_gallery_coverage,
                            episodic_gallery_margin,
                        ) = episodic_gallery_energy_ranking_loss(
                            model=model,
                            noisy=noisy,
                            timesteps=timesteps,
                            text_embed=text_embed,
                            labels=labels,
                            seen_labels=seen_tensor_choices,
                            target=features,
                            scheduler=scheduler,
                            prediction_type=prediction_type,
                            gallery_size=int(config.get("episodic_gallery_size", 12)),
                            temperature=float(config.get("episodic_gallery_temperature", 0.10)),
                        )
                        warmup_steps = max(
                            0, int(config.get("episodic_gallery_warmup_steps", 1000))
                        )
                        warmup_scale = (
                            min(1.0, float(global_step + 1) / float(warmup_steps))
                            if warmup_steps > 0
                            else 1.0
                        )
                        episodic_gallery_weight = float(
                            config.get("episodic_gallery_weight", 0.05)
                        ) * warmup_scale
                if loss_aware_main_loss is not None:
                    primitive_main_loss = loss_aware_main_loss
                else:
                    primitive_main_loss = (
                        recon * float(config.get("rec_weight", config.get("d_weight", 1.0)))
                        + hard_rank * float(config.get("tdsm_triplet_weight", config.get("t_weight", 1.0)))
                    )
                primitive_main_loss = (
                    primitive_main_loss
                    + semantic_kd * semantic_weight
                    + prototype_direction * prototype_direction_weight
                    + energy_calibration * energy_calibration_weight
                    + energy_distill_kl * energy_distill_weight
                    + energy_distill_gap * energy_distill_gap_weight
                    + primitive_uot * primitive_uot_weight
                    + ordinal_geometry * ordinal_geometry_weight
                    + timestep_gap_loss * timestep_gap_weight
                    + episodic_gallery_rank * episodic_gallery_weight
                )
                non_primitive_alignment_weight = multipos_weight + neutral_align_weight
                if non_primitive_alignment_weight > 0.0:
                    primitive_main_loss = (
                        primitive_main_loss + text_align * non_primitive_alignment_weight
                    )
                primitive_weighted_loss = (
                    text_align * primitive_multilabel_weight
                    + f1_gallery_match * f1_gallery_match_weight
                    + primitive_proto_contrast * primitive_proto_contrast_weight
                    + primitive_tmr_infonce * primitive_tmr_weight
                    + primitive_feature_nll * primitive_feature_nll_weight
                    + primitive_feature_recon * primitive_feature_recon_weight
                    + clean_tide_primitive * clean_tide_primitive_weight
                    + clean_tide_distill * clean_tide_distill_weight
                    + episodic_proto_mse * episodic_proto_mse_weight
                    + episodic_proto_nll * episodic_proto_nll_weight
                    + counterfactual_text_loss * counterfactual_text_weight
                )
                loss = primitive_main_loss + primitive_weighted_loss
            elif use_align_loss:
                # c2u_feat_align: SupCon alignment (cosine in projection space) +
                # auxiliary reconstruction loss to keep DiT grounded.
                pred, hidden_states = model(
                    noisy,
                    timesteps.to(dtype),
                    fc,
                    fl,
                    return_hidden=True,
                    hidden_layers=[align_layer],
                )
                anchor_hidden = DiT.pool_hidden(hidden_states[0])
                supcon = supcon_align_loss(
                    aux=aux,
                    hidden=anchor_hidden,
                    text_embed=text_embed,
                    labels=labels,
                    seen_labels=seen_tensor_choices,
                    tau=float(config.get("supcon_tau", 0.07)),
                    n_text_neg=int(config.get("supcon_n_neg", 47)),
                    device=device,
                    topk_hard_bank=topk_hard_bank,
                )
                recon = F.mse_loss(pred, target)
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = torch.tensor(0.0, device=device)
                infonce = supcon
                projected_infonce = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                loss = (
                    recon * float(config.get("rec_weight", 0.5))
                    + supcon * float(config.get("supcon_weight", 1.0))
                )
            else:
                pred = model(noisy, timesteps.to(dtype), fc, fl)
                pred_sample = prediction_to_sample(pred, noisy, timesteps, scheduler, prediction_type)
                recon = F.mse_loss(pred, target)
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = torch.tensor(0.0, device=device)
                infonce = torch.tensor(0.0, device=device)
                projected_infonce = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                loss = recon

            accum_count += 1
            if use_primitive_pcgrad:
                if primitive_main_loss is None or primitive_weighted_loss is None:
                    raise RuntimeError("PCGrad primitive losses were not initialized")
                (
                    primitive_grad_cosine,
                    primitive_grad_scale,
                    primitive_grad_conflict,
                ) = backward_primitive_pcgrad(
                    primitive_main_loss,
                    primitive_weighted_loss,
                    model_trainable_params,
                    aux_trainable_params,
                    grad_accum_steps,
                    primitive_pcgrad_max_ratio,
                )
            else:
                (loss / grad_accum_steps).backward()

            should_step = (
                accum_count >= grad_accum_steps
                or (batch_idx + 1 == len(train_loader))
                or (max_train_batches > 0 and batch_idx + 1 >= max_train_batches)
            )
            if should_step:
                if accum_count != grad_accum_steps:
                    scale = grad_accum_steps / max(1, accum_count)
                    for param in trainable_params:
                        if param.grad is not None:
                            param.grad.mul_(scale)
                grad_clip = float(config.get("grad_clip", 1.0))
                if use_primitive_multilabel and (
                    bool(config.get("primitive_detach_backbone", True)) or use_primitive_pcgrad
                ):
                    torch.nn.utils.clip_grad_norm_(model_trainable_params, grad_clip)
                    if aux_trainable_params:
                        torch.nn.utils.clip_grad_norm_(aux_trainable_params, grad_clip)
                    if clean_tide_trainable_params:
                        torch.nn.utils.clip_grad_norm_(clean_tide_trainable_params, grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
                lr_scheduler.step()
                if use_da_cnce_trust and da_cnce_reference_model is not None:
                    violation = float(trust_region.detach().item()) - float(
                        config.get("da_cnce_trust_delta", 0.002)
                    )
                    da_cnce_dual = min(
                        float(config.get("da_cnce_dual_max", 10.0)),
                        max(0.0, da_cnce_dual + float(config.get("da_cnce_dual_lr", 0.1)) * violation),
                    )
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)
                        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                            ema_buffer.copy_(buffer)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1

            epoch_loss += float(loss.detach().cpu())
            epoch_recon += float(recon.detach().cpu())
            epoch_easy_rank += float(easy_rank.detach().cpu())
            epoch_hard_rank += float(hard_rank.detach().cpu())
            epoch_infonce += float(infonce.detach().cpu())
            epoch_proj_infonce += float(projected_infonce.detach().cpu())
            epoch_gen_con += float(gen_con.detach().cpu())
            epoch_text_align += float(text_align.detach().cpu())
            epoch_dcr += float(dcr.detach().cpu())
            epoch_semantic_kd += float(semantic_kd.detach().cpu())
            epoch_prototype_direction += float(prototype_direction.detach().cpu())
            epoch_prototype_reliability += float(prototype_reliability.detach().cpu())
            epoch_adaptive_margin += float(adaptive_margin_mean.detach().cpu())
            epoch_margin_reliability += float(adaptive_margin_reliability.detach().cpu())
            epoch_rank_active += float(rank_active_fraction.detach().cpu())
            epoch_noise_gap += float(noise_gap_mean.detach().cpu())
            epoch_noise_gap_std += float(noise_gap_std.detach().cpu())
            epoch_rank_agreement += float(rank_agreement_coverage.detach().cpu())
            epoch_timestep_gap_loss += float(timestep_gap_loss.detach().cpu())
            epoch_timestep_gap_coverage += float(timestep_gap_coverage.detach().cpu())
            epoch_timestep_gap_anchor += float(timestep_gap_anchor_mean.detach().cpu())
            epoch_timestep_gap_partner += float(timestep_gap_partner_mean.detach().cpu())
            epoch_loss_aware_entropy += float(loss_aware_entropy.detach().cpu())
            epoch_loss_aware_importance += float(loss_aware_importance.mean().detach().cpu())
            epoch_loss_aware_active += loss_aware_active
            epoch_adaptive_timestep += float(adaptive_timestep_mean.detach().cpu())
            epoch_teacher_gap += float(teacher_gap_mean.detach().cpu())
            epoch_teacher_valid += float(teacher_valid_fraction.detach().cpu())
            epoch_ambiguity_rank += float(ambiguity_rank_loss.detach().cpu())
            epoch_ambiguity_coverage += float(ambiguity_coverage.detach().cpu())
            epoch_ambiguity_active += float(ambiguity_active_fraction.detach().cpu())
            epoch_multipos_snr += float(multipos_snr_mean.detach().cpu())
            epoch_neutral_negative_weight += float(neutral_negative_weight.detach().cpu())
            epoch_primitive_density += float(primitive_target_density.detach().cpu())
            epoch_clean_tide_distill += float(clean_tide_distill.detach().cpu())
            epoch_clean_tide_primitive += float(clean_tide_primitive.detach().cpu())
            epoch_clean_tide_cosine += float(clean_tide_cosine.detach().cpu())
            epoch_primitive_pu_active += float(primitive_pu_active.detach().cpu())
            epoch_primitive_pu_corrected_negative += float(
                primitive_pu_corrected_negative.detach().cpu()
            )
            epoch_primitive_proto_contrast += float(primitive_proto_contrast.detach().cpu())
            epoch_primitive_proto_negative_weight += float(
                primitive_proto_negative_weight.detach().cpu()
            )
            epoch_primitive_tmr_infonce += float(primitive_tmr_infonce.detach().cpu())
            epoch_primitive_tmr_filtered += float(primitive_tmr_filtered.detach().cpu())
            epoch_primitive_feature_nll += float(primitive_feature_nll.detach().cpu())
            epoch_primitive_feature_recon += float(primitive_feature_recon.detach().cpu())
            epoch_primitive_feature_variance += float(primitive_feature_variance.detach().cpu())
            epoch_episodic_proto_mse += float(episodic_proto_mse.detach().cpu())
            epoch_episodic_proto_nll += float(episodic_proto_nll.detach().cpu())
            epoch_episodic_gallery_rank += float(episodic_gallery_rank.detach().cpu())
            epoch_episodic_gallery_coverage += float(episodic_gallery_coverage.detach().cpu())
            epoch_episodic_gallery_margin += float(episodic_gallery_margin.detach().cpu())
            epoch_f1_gallery_match += float(f1_gallery_match.detach().cpu())
            epoch_f1_csv_gate += float(f1_csv_gate.detach().cpu())
            epoch_counterfactual_text_loss += float(counterfactual_text_loss.detach().cpu())
            epoch_counterfactual_text_gap += float(counterfactual_text_gap.detach().cpu())
            epoch_energy_calibration += float(energy_calibration.detach().cpu())
            epoch_energy_distill_kl += float(energy_distill_kl.detach().cpu())
            epoch_energy_distill_gap += float(energy_distill_gap.detach().cpu())
            epoch_energy_distill_agreement += float(energy_distill_agreement.detach().cpu())
            epoch_primitive_uot += float(primitive_uot.detach().cpu())
            epoch_primitive_uot_target_l1 += float(primitive_uot_target_l1.detach().cpu())
            epoch_ordinal_geometry += float(ordinal_geometry.detach().cpu())
            epoch_ordinal_geometry_coverage += float(ordinal_geometry_coverage.detach().cpu())
            epoch_primitive_grad_cosine += primitive_grad_cosine
            epoch_primitive_grad_scale += primitive_grad_scale
            epoch_primitive_grad_conflict += primitive_grad_conflict
            epoch_batches += 1

            if global_step >= total_steps or (
                max_train_batches > 0 and batch_idx + 1 >= max_train_batches
            ):
                break

        denom = max(1, epoch_batches)
        timestep_selection_total = max(1, sum(epoch_timestep_counts))
        timestep_selection_summary = ",".join(
            f"{timestep}:{count / timestep_selection_total:.3f}"
            for timestep, count in zip(adaptive_timestep_candidates, epoch_timestep_counts)
        )
        train_summary = [
            f"Iter[{global_step}/{total_steps}]",
            f"Loss={epoch_loss / denom:.6f}",
            f"Recon={epoch_recon / denom:.6f}",
            f"LR={optimizer.param_groups[0]['lr']:.2e}",
        ]
        is_tdsm_objective = loss_mode.startswith("tdsm_") or loss_mode in {
            "x0_triplet",
            "tide_x0_contrastive",
            "tide_x0_rank",
        }
        if is_tdsm_objective:
            train_summary.extend(
                [
                    f"Rank={epoch_hard_rank / denom:.6f}",
                    f"RankActive={epoch_rank_active / denom:.4f}",
                ]
            )
        elif use_c2u_loss:
            train_summary.extend(
                [
                    f"EasyRank={epoch_easy_rank / denom:.6f}",
                    f"HardRank={epoch_hard_rank / denom:.6f}",
                    f"InfoNCE={epoch_infonce / denom:.6f}",
                ]
            )
        if use_primitive_multilabel:
            train_summary.extend(
                [
                    f"Primitive={epoch_text_align / denom:.6f}",
                    f"PrimitiveDensity={epoch_primitive_density / denom:.4f}",
                ]
            )
        if use_frozen_gallery_match:
            train_summary.extend(
                [
                    f"F1Match={epoch_f1_gallery_match / denom:.6f}",
                    f"F1Gate={epoch_f1_csv_gate / denom:.4f}",
                ]
            )
        if use_energy_calibration:
            train_summary.append(f"EnergyCal={epoch_energy_calibration / denom:.6f}")
        if use_multitime_energy_distill:
            train_summary.extend(
                [
                    f"EnergyKL={epoch_energy_distill_kl / denom:.6f}",
                    f"EnergyGap={epoch_energy_distill_gap / denom:.6f}",
                ]
            )
        if use_loss_aware_timestep:
            train_summary.extend(
                [
                    f"LASEntropy={epoch_loss_aware_entropy / denom:.6f}",
                    f"LASActive={epoch_loss_aware_active / denom:.3f}",
                    f"TSelect={timestep_selection_summary}",
                ]
            )
        log(train_log, "\t".join(train_summary))

        if eval_during_training and (epoch + 1) % int(config.get("eval_epoch", 1)) == 0:
            eval_model = ema_model if ema_model is not None else model
            eval_aux = aux
            metrics = evaluate_zsl(eval_model, test_loader, scheduler, text_embed, unseen_labels, config, device, dtype, aux=eval_aux, precision=eval_precision, bridge_ctx=eval_bridge_ctx, analogy_ctx=eval_analogy_ctx, primitive_ctx=primitive_context, episodic_proto_ctx=episodic_proto_context, clean_tide=clean_tide)
            log(
                val_log,
                (
                    f"ZSL {eval_split_name} Acc: {metrics['accuracy']:.6f}\tEpoch: {epoch + 1}\t"
                    f"Samples: {metrics['num_samples']}\tSpeed: {metrics['samples_per_sec']:.4f}\t"
                    f"EvalT: {metrics['eval_timesteps']}\tEvalNoise: {metrics['eval_num_noise']}\t"
                    f"EvalSeed: {metrics['eval_noise_seed']}\tEvalNoiseMode: {metrics['eval_noise_mode']}\t"
                    f"EMA: {use_ema}\t"
                    f"EvalSpace: {metrics['eval_distance_space']}\t"
                    f"CFG: {metrics.get('cfg_scale', 1.0):.1f}\tMahal: {metrics.get('use_mahal', False)}"
                    + (
                        f"\tBaseAcc: {metrics.get('base_accuracy', metrics['accuracy']):.6f}"
                        f"\tProjAlpha: {metrics.get('proj_effective_alpha', 0.0):.6f}"
                        if bool(config.get("eval_proj_late_fusion", False))
                        else ""
                    )
                    + (
                        f"\tBaseAcc: {metrics.get('base_accuracy', metrics['accuracy']):.6f}"
                        f"\tPrimitiveAlpha: {metrics.get('primitive_effective_alpha', 0.0):.6f}"
                        f"\tPrimitiveReliability: {metrics.get('primitive_candidate_reliability', 1.0):.6f}"
                        if bool(config.get("eval_primitive_late_fusion", False))
                        else ""
                    )
                    + (f"\tProjAcc: {metrics['proj_accuracy']:.6f}" if metrics.get("proj_accuracy", -1) >= 0 else "")
                    + (
                        f"\tF1MatchAcc: {metrics['f1_gallery_match_accuracy']:.6f}"
                        f"\tF1Alpha: {metrics.get('f1_gallery_match_effective_alpha', 0.0):.6f}"
                        if metrics.get("f1_gallery_match_accuracy", -1) >= 0
                        else ""
                    )
                    + (
                        f"\tPrimitiveAcc: {metrics['primitive_accuracy']:.6f}"
                        if metrics.get("primitive_accuracy", -1) >= 0
                        else ""
                    )
                    + (
                        f"\tCleanTidePrimitiveAcc: {metrics['clean_tide_primitive_accuracy']:.6f}"
                        f"\tCleanTideAlpha: {metrics.get('clean_tide_effective_alpha', 0.0):.6f}"
                        if metrics.get("clean_tide_primitive_accuracy", -1) >= 0
                        else ""
                    )
                    + (
                        f"\tFeatureDistAcc: {metrics['feature_distribution_accuracy']:.6f}"
                        f"\tFeatureDistAlpha: {metrics.get('feature_distribution_effective_alpha', 0.0):.6f}"
                        if metrics.get("feature_distribution_accuracy", -1) >= 0
                        else ""
                    )
                    + (
                        f"\tEpisodicProtoAcc: {metrics['episodic_proto_accuracy']:.6f}"
                        f"\tEpisodicProtoSeenAcc: {metrics.get('episodic_proto_seen_accuracy', 0.0):.6f}"
                        f"\tEpisodicProtoRel: {metrics.get('episodic_proto_reliability', 0.0):.6f}"
                        f"\tEpisodicProtoAlpha: {metrics.get('episodic_proto_effective_alpha', 0.0):.6f}"
                        if metrics.get("episodic_proto_accuracy", -1) >= 0
                        else ""
                    )
                    + (
                        f"\tOT: uniform_sinkhorn"
                        f"\tOTEps: {metrics.get('ot_epsilon', 0.0):.4f}"
                        f"\tOTIters: {metrics.get('ot_iterations', 0)}"
                        f"\tOTMargErr: {metrics.get('ot_marginal_error', 0.0):.3e}"
                        if metrics.get("ot_enabled", False)
                        else ""
                    )
                ),
            )
            save_eval_artifacts(work_dir, metrics, "latest")
            can_update_best = metrics["num_samples"] >= min_best_eval_samples
            if can_update_best and metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                epochs_without_improvement = 0
                save_eval_artifacts(work_dir, metrics, "best")
                save_checkpoint(
                    work_dir / "best",
                    model,
                    optimizer,
                    lr_scheduler,
                    {"epoch": epoch + 1, "global_step": global_step, "best_zsl_acc": best_acc},
                    aux=aux,
                    clean_tide=clean_tide,
                    ema_model=ema_model,
                    save_ema_as_model=ema_model is not None,
                )
                log(
                    val_log,
                    f"Best {eval_split_name} Acc: {best_acc:.6f}\tBest Epoch: {epoch + 1}",
                )
            elif can_update_best:
                epochs_without_improvement += 1
                if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                    should_early_stop = True
                    log(
                        train_log,
                        (
                            f"Early stop: no new {eval_split_name} best for "
                            f"{epochs_without_improvement} evaluation epochs"
                        ),
                    )

        save_checkpoint(
            work_dir / "latest",
            model,
            optimizer,
            lr_scheduler,
            {"epoch": epoch + 1, "global_step": global_step, "best_zsl_acc": best_acc},
            aux=aux,
            clean_tide=clean_tide,
            ema_model=ema_model,
            save_ema_as_model=False,
        )
        if global_step >= total_steps:
            break
        if should_early_stop:
            break

    train_log.close()
    val_log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train feature-space C2U x-prediction ZSL.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed without duplicating an otherwise identical config.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--text-llm-path", default=None, help="Override the second-view LLM prompt file.")
    parser.add_argument("--clip-text-feature-path", default=None, help="Override the cached concatenated SD2-CLIP token feature file.")
    parser.add_argument(
        "--adaptive-timestep-teacher-checkpoint",
        default=None,
        help="Frozen teacher checkpoint used by timestep policy distillation.",
    )
    parser.add_argument("--work-dir", default=None, help="Override config work_dir, useful for isolated smoke runs.")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=None, help="Override total optimizer steps.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap batches per epoch; intended for a complete-data smoke run.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--disable-eval-during-training",
        action="store_true",
        help="Do not evaluate on the configured test split during training.",
    )
    parser.add_argument("--eval-tag", default="eval_only")
    parser.add_argument("--loss-mode", default=None, help="Override loss_mode from the YAML config.")
    parser.add_argument(
        "--tdsm-triplet-weight",
        type=float,
        default=None,
        help="Override the TDSM contrastive/triplet loss coefficient without changing the YAML config.",
    )
    parser.add_argument(
        "--tdsm-triplet-margin",
        type=float,
        default=None,
        help="Override the TDSM triplet/ranking margin without changing the YAML config.",
    )
    parser.add_argument(
        "--primitive-multilabel-weight",
        type=float,
        default=None,
        help="Override the C10 primitive multi-label loss coefficient without changing the YAML config.",
    )
    parser.add_argument(
        "--eval-primitive-late-alpha",
        type=float,
        default=None,
        help="Override the C10 primitive late-fusion alpha at evaluation time.",
    )
    parser.add_argument(
        "--eval-primitive-late-topk",
        type=int,
        default=None,
        help="Override the number of A0 candidates re-ranked by C10 at evaluation time.",
    )
    parser.add_argument(
        "--eval-primitive-negative-weight",
        type=float,
        default=None,
        help="Override the C10 primitive negative-evidence score weight at evaluation time.",
    )
    parser.add_argument(
        "--disable-self-attention",
        action="store_true",
        help="Disable skeleton-token self-attention for the fine-grained TIDE-DiT ablation.",
    )
    parser.add_argument(
        "--disable-cross-attention",
        action="store_true",
        help="Disable skeleton-to-text cross-attention for the fine-grained TIDE-DiT ablation.",
    )
    parser.add_argument(
        "--disable-global-text",
        action="store_true",
        help="Disable pooled global-text modulation while retaining token-level cross-attention.",
    )
    parser.add_argument(
        "--disable-f1",
        action="store_true",
        help="Disable F1 frozen-gallery training and evaluation; use the base TDSM triplet objective.",
    )
    parser.add_argument(
        "--disable-c10",
        action="store_true",
        help="Disable C10 primitive-score evaluation and use the base TDSM objective.",
    )
    parser.add_argument(
        "--disable-transductive",
        action="store_true",
        help="Disable the separate transductive score calibration during evaluation.",
    )
    parser.add_argument(
        "--disable-c21",
        action="store_true",
        help="Legacy alias for --disable-feature-graph.",
    )
    parser.add_argument(
        "--enable-feature-graph",
        action="store_true",
        dest="eval_c21",
        help="Enable frozen mutual-kNN feature-graph refinement during evaluation.",
    )
    parser.add_argument(
        "--disable-feature-graph",
        action="store_true",
        dest="disable_c21",
        help="Disable frozen mutual-kNN feature-graph refinement during evaluation.",
    )
    parser.add_argument(
        "--inductive-eval",
        action="store_true",
        help=(
            "Use per-sample inductive evaluation: disable all whole-test-set "
            "calibration, transport, and feature-graph refinement."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after this many evaluation epochs without a new best; zero disables it.",
    )
    parser.add_argument(
        "--eval-primitive-score-mode",
        choices=["c10", "hard_negative", "positive_only", "pu_positive", "pu_soft"],
        default=None,
        help="Override primitive evaluation scoring without changing the YAML config.",
    )
    parser.add_argument(
        "--primitive-pu-positive-probability",
        type=float,
        default=None,
        help="Posterior primitive probability for a token explicitly present in a class prompt.",
    )
    parser.add_argument(
        "--eval-primitive-risk-gate-path",
        default=None,
        help="Path to a frozen primitive-risk gate JSON fitted only on pseudo-unseen data.",
    )
    parser.add_argument(
        "--eval-primitive-utility-gate-path",
        default=None,
        help="Path to a frozen utility-calibrated primitive gate fitted only on pseudo-unseen data.",
    )
    parser.add_argument(
        "--c23-geometry-metric-path",
        default=None,
        help="Path to the frozen all-seen C23 text-metric .npz file.",
    )
    parser.add_argument(
        "--c23-ordinal-weight",
        type=float,
        default=None,
        help="Override the C23 ordinal-energy loss weight.",
    )
    parser.add_argument(
        "--c24-reliability-path",
        default=None,
        help="Path to a frozen C24 per-class energy-order reliability JSON file.",
    )
    parser.add_argument(
        "--eval-feature-graph-alpha",
        type=float,
        default=None,
        help="Override eval_feature_graph_alpha without changing the YAML config.",
    )
    parser.add_argument(
        "--eval-c21",
        action="store_true",
        help="Enable C21 frozen feature-graph score refinement at evaluation time.",
    )
    parser.add_argument(
        "--eval-feature-graph-k",
        type=int,
        default=None,
        help="Override the C21 mutual-neighbor count at evaluation time.",
    )
    parser.add_argument(
        "--eval-feature-graph-tau",
        type=float,
        default=None,
        help="Override the C21 graph similarity temperature at evaluation time.",
    )
    parser.add_argument(
        "--eval-feature-graph-iters",
        type=int,
        default=None,
        help="Override the C21 propagation iteration count at evaluation time.",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Override conditional-energy evaluation timesteps without changing the YAML config.",
    )
    parser.add_argument(
        "--eval-reverse-x0-timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Enable E1 deterministic reverse-x0 evaluation with a strictly descending timestep path.",
    )
    parser.add_argument(
        "--eval-primitive-timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Override primitive-score timesteps; set this with --eval-timesteps for a synchronized fixed-t probe.",
    )
    parser.add_argument(
        "--eval-d6-evidence",
        action="store_true",
        help="Enable D6 calibrated multi-timestep conditional-evidence integration.",
    )
    parser.add_argument(
        "--eval-adaptive-timestep",
        action="store_true",
        help="Enable D9 per-sample, label-free adaptive timestep energy fusion.",
    )
    parser.add_argument(
        "--eval-adaptive-timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Candidate timesteps for D9 adaptive evidence fusion.",
    )
    parser.add_argument(
        "--eval-adaptive-temperature",
        type=float,
        default=None,
        help="Softmax temperature for D9 per-sample timestep evidence weights.",
    )
    parser.add_argument("--episodic-gallery-size", type=int, default=None)
    parser.add_argument("--episodic-gallery-interval", type=int, default=None)
    parser.add_argument("--episodic-gallery-start-step", type=int, default=None)
    parser.add_argument("--episodic-gallery-weight", type=float, default=None)
    parser.add_argument("--episodic-gallery-temperature", type=float, default=None)
    parser.add_argument(
        "--eval-d6-timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timestep gallery for D6 evidence integration.",
    )
    parser.add_argument(
        "--eval-d6-weight-path",
        default=None,
        help="Frozen seen-only D6 weight JSON produced by fit_d6_evidence_weights.py.",
    )
    parser.add_argument(
        "--eval-semi-uot",
        action="store_true",
        help="Apply row-constrained, target-unbalanced OT after C21 score refinement.",
    )
    parser.add_argument(
        "--eval-semi-uot-epsilon",
        type=float,
        default=None,
        help="Override the entropic temperature for evaluation semi-unbalanced OT.",
    )
    parser.add_argument(
        "--eval-semi-uot-target-tau",
        type=float,
        default=None,
        help="Override the target-marginal KL weight for evaluation semi-unbalanced OT.",
    )
    parser.add_argument(
        "--eval-semi-uot-iterations",
        type=int,
        default=None,
        help="Override the Sinkhorn iteration count for evaluation semi-unbalanced OT.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.adaptive_timestep_teacher_checkpoint is not None:
        config["adaptive_timestep_teacher_checkpoint"] = args.adaptive_timestep_teacher_checkpoint
    if args.work_dir:
        config["work_dir"] = args.work_dir
    if args.text_llm_path:
        config["text_llm_path"] = args.text_llm_path
    if args.clip_text_feature_path:
        config["clip_text_feature_path"] = args.clip_text_feature_path
    if args.disable_eval_during_training:
        config["eval_during_training"] = False
    if args.num_iter is not None:
        if args.num_iter < 1:
            parser.error("--num-iter must be positive")
        config["num_iter"] = args.num_iter
    if args.max_train_batches is not None:
        if args.max_train_batches < 1:
            parser.error("--max-train-batches must be positive")
        config["max_train_batches"] = args.max_train_batches
    if args.loss_mode is not None:
        config["loss_mode"] = args.loss_mode
    if args.tdsm_triplet_weight is not None:
        if args.tdsm_triplet_weight < 0.0:
            parser.error("--tdsm-triplet-weight must be non-negative")
        config["tdsm_triplet_weight"] = args.tdsm_triplet_weight
    if args.tdsm_triplet_margin is not None:
        if args.tdsm_triplet_margin < 0.0:
            parser.error("--tdsm-triplet-margin must be non-negative")
        config["tdsm_triplet_margin"] = args.tdsm_triplet_margin
    if args.primitive_multilabel_weight is not None:
        if args.primitive_multilabel_weight < 0.0:
            parser.error("--primitive-multilabel-weight must be non-negative")
        config["primitive_multilabel_weight"] = args.primitive_multilabel_weight
    if args.eval_primitive_late_alpha is not None:
        if not 0.0 <= args.eval_primitive_late_alpha <= 1.0:
            parser.error("--eval-primitive-late-alpha must be in [0, 1]")
        config["eval_primitive_late_alpha"] = args.eval_primitive_late_alpha
    if args.eval_primitive_late_topk is not None:
        if args.eval_primitive_late_topk < 1:
            parser.error("--eval-primitive-late-topk must be positive")
        config["eval_primitive_late_topk"] = args.eval_primitive_late_topk
    if args.eval_primitive_negative_weight is not None:
        if args.eval_primitive_negative_weight < 0.0:
            parser.error("--eval-primitive-negative-weight must be non-negative")
        config["eval_primitive_negative_weight"] = args.eval_primitive_negative_weight
    if args.disable_self_attention:
        config["enable_self_attention"] = False
    if args.disable_cross_attention:
        config["enable_cross_attention"] = False
    if args.disable_global_text:
        config["enable_global_text"] = False
    if args.disable_f1:
        config["loss_mode"] = "tdsm_x0_triplet"
        config["f1_match_weight"] = 0.0
        config["eval_f1_gallery_match"] = False
    if args.disable_c10:
        config["loss_mode"] = "tdsm_x0_triplet"
        config["eval_primitive_score"] = False
    if args.disable_transductive:
        config["eval_transductive"] = False
    if args.inductive_eval:
        # These operations aggregate statistics or neighbourhoods over the full
        # unlabeled test split. Per-sample row normalization remains enabled.
        config["eval_a3_calib"] = False
        config["eval_transductive"] = False
        config["eval_class_balance"] = False
        config["eval_feature_graph"] = False
        config["eval_transductive_ot"] = False
        config["eval_semi_uot"] = False
        config["eval_primitive_a3_calib"] = False
    if args.early_stop_patience is not None:
        if args.early_stop_patience < 0:
            parser.error("--early-stop-patience must be non-negative")
        config["early_stop_patience"] = args.early_stop_patience
    if args.eval_primitive_score_mode is not None:
        config["eval_primitive_score_mode"] = args.eval_primitive_score_mode
    if args.primitive_pu_positive_probability is not None:
        if not 0.0 <= args.primitive_pu_positive_probability <= 1.0:
            parser.error("--primitive-pu-positive-probability must be in [0, 1]")
        config["primitive_pu_positive_probability"] = args.primitive_pu_positive_probability
    if args.eval_primitive_risk_gate_path is not None:
        config["eval_primitive_risk_gate_path"] = args.eval_primitive_risk_gate_path
    if args.eval_primitive_utility_gate_path is not None:
        config["eval_primitive_utility_gate_path"] = args.eval_primitive_utility_gate_path
    if args.c23_geometry_metric_path is not None:
        config["c23_geometry_metric_path"] = args.c23_geometry_metric_path
    if args.c23_ordinal_weight is not None:
        if args.c23_ordinal_weight < 0.0:
            parser.error("--c23-ordinal-weight must be non-negative")
        config["c23_ordinal_weight"] = args.c23_ordinal_weight
    if args.c24_reliability_path is not None:
        config["c24_reliability_path"] = args.c24_reliability_path
    if args.eval_feature_graph_alpha is not None:
        if not 0.0 <= args.eval_feature_graph_alpha <= 1.0:
            parser.error("--eval-feature-graph-alpha must be in [0, 1]")
        config["eval_feature_graph_alpha"] = args.eval_feature_graph_alpha
    if args.eval_c21:
        config["eval_feature_graph"] = True
    if args.disable_c21:
        config["eval_feature_graph"] = False
    if args.eval_feature_graph_k is not None:
        if args.eval_feature_graph_k < 1:
            parser.error("--eval-feature-graph-k must be positive")
        config["eval_feature_graph_k"] = args.eval_feature_graph_k
    if args.eval_feature_graph_tau is not None:
        if args.eval_feature_graph_tau <= 0.0:
            parser.error("--eval-feature-graph-tau must be positive")
        config["eval_feature_graph_tau"] = args.eval_feature_graph_tau
    if args.eval_feature_graph_iters is not None:
        if args.eval_feature_graph_iters < 1:
            parser.error("--eval-feature-graph-iters must be positive")
        config["eval_feature_graph_iters"] = args.eval_feature_graph_iters
    for option_name, values, config_key in (
        ("--eval-timesteps", args.eval_timesteps, "eval_timesteps"),
        ("--eval-primitive-timesteps", args.eval_primitive_timesteps, "eval_primitive_timesteps"),
        ("--eval-reverse-x0-timesteps", args.eval_reverse_x0_timesteps, "eval_reverse_x0_timesteps"),
    ):
        if values is not None:
            if not values or min(values) < 0 or max(values) >= int(config["num_steps"]):
                parser.error(f"{option_name} must contain valid scheduler timesteps")
            config[config_key] = [int(value) for value in values]
    if args.eval_d6_evidence:
        config["eval_d6_evidence"] = True
    if args.eval_adaptive_timestep:
        config["eval_adaptive_timestep"] = True
    if args.eval_adaptive_timesteps is not None:
        config["eval_adaptive_timesteps"] = args.eval_adaptive_timesteps
    if args.eval_adaptive_temperature is not None:
        if args.eval_adaptive_temperature <= 0.0:
            parser.error("--eval-adaptive-temperature must be positive")
        config["eval_adaptive_temperature"] = args.eval_adaptive_temperature
    for option_name, value, config_key, minimum, strict in (
        ("--episodic-gallery-size", args.episodic_gallery_size, "episodic_gallery_size", 2, False),
        ("--episodic-gallery-interval", args.episodic_gallery_interval, "episodic_gallery_interval", 1, False),
        ("--episodic-gallery-start-step", args.episodic_gallery_start_step, "episodic_gallery_start_step", 0, False),
        ("--episodic-gallery-weight", args.episodic_gallery_weight, "episodic_gallery_weight", 0.0, False),
        ("--episodic-gallery-temperature", args.episodic_gallery_temperature, "episodic_gallery_temperature", 0.0, True),
    ):
        if value is not None:
            if (strict and value <= minimum) or (not strict and value < minimum):
                parser.error(f"{option_name} must be {'positive' if strict else f'at least {minimum}'}")
            config[config_key] = value
    if args.eval_d6_timesteps is not None:
        if (
            not args.eval_d6_timesteps
            or min(args.eval_d6_timesteps) < 0
            or max(args.eval_d6_timesteps) >= int(config["num_steps"])
            or len(set(args.eval_d6_timesteps)) != len(args.eval_d6_timesteps)
        ):
            parser.error("--eval-d6-timesteps must contain unique valid scheduler timesteps")
        config["eval_d6_timesteps"] = [int(value) for value in args.eval_d6_timesteps]
    if args.eval_d6_weight_path is not None:
        config["eval_d6_weight_path"] = args.eval_d6_weight_path
    if args.eval_semi_uot:
        config["eval_semi_uot"] = True
    if args.eval_semi_uot_epsilon is not None:
        if args.eval_semi_uot_epsilon <= 0.0:
            parser.error("--eval-semi-uot-epsilon must be positive")
        config["eval_semi_uot_epsilon"] = args.eval_semi_uot_epsilon
    if args.eval_semi_uot_target_tau is not None:
        if args.eval_semi_uot_target_tau < 0.0:
            parser.error("--eval-semi-uot-target-tau must be non-negative")
        config["eval_semi_uot_target_tau"] = args.eval_semi_uot_target_tau
    if args.eval_semi_uot_iterations is not None:
        if args.eval_semi_uot_iterations < 1:
            parser.error("--eval-semi-uot-iterations must be at least one")
        config["eval_semi_uot_iterations"] = args.eval_semi_uot_iterations
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
