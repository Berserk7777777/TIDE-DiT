from __future__ import annotations

import argparse
import copy
import csv
import json
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
from model.dit_finegrained import DiTFineGrained  # noqa: E402
from model.dit_textgate import DiTTextGate  # noqa: E402


class C2UAuxHeads(nn.Module):
    def __init__(self, hidden_size: int, text_size: int, feature_size: int = 256, proj_size: int = 256):
        super().__init__()
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_size),
        )
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, proj_size),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, proj_size),
        )

    def project_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.hidden_proj(x), dim=-1)

    def project_feature(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.feature_proj(x), dim=-1)

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(x), dim=-1)


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


def make_loader(config: dict, split_key: str, shuffle: bool, max_samples: int = 0) -> DataLoader:
    args = dict(config[f"{split_key}_feeder_args"])
    args["path"] = str(repo_path(args["path"]))
    dataset = FeatureFeeder(**args)
    if max_samples and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
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
        semantic_file = config.get("semantic_feature_file", "concat.npy")
        semantic_path = semantic_dir / semantic_file
        text_embed = np.load(semantic_path).astype(np.float32)
        label_file = semantic_dir / config.get("semantic_label_file", "labels.npy")
        if label_file.exists():
            labels = np.load(label_file).astype(np.int64)
            if labels.shape[0] == text_embed.shape[0] and not np.array_equal(labels, np.arange(text_embed.shape[0])):
                order = np.argsort(labels)
                text_embed = text_embed[order]
        return torch.from_numpy(text_embed).to(device=device, dtype=dtype)

    cache_path_value = config.get("clip_text_feature_path")
    if cache_path_value:
        cache_path = repo_path(cache_path_value)
        if cache_path.exists():
            text_embed = np.load(cache_path).astype(np.float32)
            text_embed = select_clip_source(text_embed)
            return torch.from_numpy(text_embed).to(device=device, dtype=dtype)

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
    llm_path = repo_path(config.get("text_llm_path", "data/class_lists/ntu60_llm.txt"))
    df = pd.read_csv(class_list_path)
    csv_prompts = df["label"].values.tolist()
    llm_prompts = llm_path.read_text(encoding="utf-8").splitlines()

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
    return select_clip_source(text_embed)


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


def build_seen_bridge_context(
    config: dict,
    unseen_labels: np.ndarray,
    text_embed: torch.Tensor,
    device: torch.device,
) -> dict | None:
    """Precompute everything needed for seen-class KNN bridge scoring.

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
        "x_seen": torch.from_numpy(x_seen).to(device=device),         # (M, feat_dim)
        "y_seen_row": torch.tensor(
            [seen_to_row[int(c)] for c in y_seen.tolist()], device=device, dtype=torch.long
        ),                                                            # (M,) -> row into text_bridge
        "text_bridge": torch.from_numpy(text_bridge).to(device=device),  # (num_seen_classes_present, num_unseen)
        "k": int(config.get("eval_bridge_k", 10)),
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
    targets = torch.zeros((labels.shape[0],), device=labels.device, dtype=torch.long)
    return F.cross_entropy(logits, targets), distance_matrix


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
    ema_model: DiT | None = None,
    save_ema_as_model: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = ema_model if save_ema_as_model and ema_model is not None else model
    metadata = dict(metadata)
    if ema_model is not None:
        metadata["use_ema"] = True
        metadata["saved_model"] = "ema" if save_ema_as_model else "raw"
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "aux": aux.state_dict() if aux is not None else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "metadata": metadata,
        },
        path / "checkpoint.pt",
    )
    (path / "training_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(
    path: Path,
    model: DiT,
    optimizer=None,
    lr_scheduler=None,
    aux: C2UAuxHeads | None = None,
    ema_model: DiT | None = None,
) -> dict:
    state = torch.load(path / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    if ema_model is not None:
        ema_state = state.get("ema_model")
        ema_model.load_state_dict(ema_state if ema_state is not None else state["model"])
    if aux is not None and state.get("aux") is not None:
        aux.load_state_dict(state["aux"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return state.get("metadata", {})


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
) -> dict:
    model.eval()
    if aux is not None:
        aux.eval()
    prediction_type = str(config.get("prediction_type", "sample"))
    loss_mode = str(config.get("loss_mode", "c2u_feat")).lower()
    eval_distance_space = str(config.get("eval_distance_space", "x0")).lower()
    if eval_distance_space not in {"x0", "sample", "prediction", "native"}:
        raise ValueError(f"Unsupported eval_distance_space: {eval_distance_space}")
    use_native_eval_distance = eval_distance_space in {"prediction", "native"}
    noise_count = int(config.get("eval_num_noise", config.get("num_noise", 1)))
    eval_seed = config.get("eval_noise_seed", None)
    eval_seed = None if eval_seed is None else int(eval_seed)
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
    label_to_idx = {int(label): idx for idx, label in enumerate(unseen_labels)}
    score_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    start_time = time.time()

    eval_align_mode = loss_mode == "c2u_feat_align" and aux is not None
    eval_also_proj_score = (
        bool(config.get("eval_also_proj_score", False))
        and aux is not None
        and not eval_align_mode
    )

    # Pre-compute projection-space ingredients outside the loop (no bsz yet).
    neutral_fc_proj: torch.Tensor | None = None
    q_t_proj: list[torch.Tensor] = []
    proj_t_values: list[int] = []
    proj_noise_count: int = 1
    proj_score_chunks: list[torch.Tensor] = []
    mahal_score_chunks: list[torch.Tensor] = []
    bridge_score_chunks: list[torch.Tensor] = []
    analogy_score_chunks: list[torch.Tensor] = []
    collect_both: bool = False  # set per-batch inside the loop, initialized here for scope

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

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(loader, desc="zsl eval")):
            features = features.to(device=device, dtype=dtype).unsqueeze(1)
            labels = labels.to(device=device, dtype=torch.long)
            bsz = features.shape[0]
            scores = torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
            proj_scores = torch.zeros((bsz, len(unseen_labels)), device=device, dtype=torch.float32)
            if eval_also_proj_score and neutral_fc_proj is None:
                # Store as (1, ...) — expand per-batch to handle variable last-batch size.
                neutral_fc_proj = pooled_text_embed(text_embed).mean(0, keepdim=True)
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
                        if eval_seed is None:
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

                for t_idx, t_value in enumerate(t_values):
                    t_float = torch.ones((bsz,), device=device, dtype=dtype) * int(t_value)
                    t_long = t_float.long()
                    for noise_idx in range(noise_count):
                        if eval_seed is None:
                            noise = torch.randn_like(features)
                        else:
                            noise_seed = eval_seed + batch_idx * 1000003 + t_idx * 1009 + noise_idx
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

                        if cfg_scale > 1.0:
                            pred_null = model(noisy, t_float, null_fc_cfg, None)

                        for col, label in enumerate(unseen_labels):
                            label_tensor = torch.full((bsz,), int(label), device=device, dtype=torch.long)
                            fc, fl = split_text_condition(text_embed, label_tensor)
                            pred = model(noisy, t_float, fc, fl)
                            if cfg_scale > 1.0:
                                pred = (1.0 + cfg_scale) * pred - cfg_scale * pred_null
                            if use_native_eval_distance:
                                pred_sample = None
                                mse_dist = reconstruction_distance(pred, native_target)
                            else:
                                pred_sample = prediction_to_sample(
                                    pred,
                                    noisy,
                                    t_long,
                                    scheduler,
                                    prediction_type,
                                )
                                mse_dist = reconstruction_distance(pred_sample, features)
                            if collect_both:
                                scores[:, col] += mse_dist
                                mahal_scores[:, col] += whitened_distance(pred_sample, features, precision)
                            elif use_mahal:
                                scores[:, col] += whitened_distance(pred_sample, features, precision)
                            else:
                                scores[:, col] += mse_dist

                denom_s = max(1, noise_count * len(t_values))
                scores = scores / denom_s
                if collect_both:
                    mahal_scores = mahal_scores / denom_s

                # Projection-space secondary scoring (eval_also_proj_score: true).
                # Uses neutral-conditioned hidden state → Proj_s(q_s) vs Proj_t(q_t_c).
                if eval_also_proj_score:
                    for t_idx2, t_value2 in enumerate(proj_t_values):
                        t_float2 = torch.ones((bsz,), device=device, dtype=dtype) * int(t_value2)
                        t_long2 = t_float2.long()
                        for noise_idx2 in range(proj_noise_count):
                            if eval_seed is None:
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

    # Mixed MSE + Mahalanobis scoring: z-score each independently, then blend.
    # alpha=1.0 → pure Mahal, alpha=0.0 → pure MSE, 0<alpha<1 → blend.
    if use_mahal and 0.0 < float(config.get("eval_mahal_alpha", 1.0)) < 1.0 and mahal_score_chunks:
        alpha = float(config.get("eval_mahal_alpha", 1.0))
        all_mahal = torch.cat(mahal_score_chunks, dim=0)
        # z-score each independently so they are on the same scale before blending
        def _zscore(s: torch.Tensor) -> torch.Tensor:
            return (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-6)
        all_scores = (1.0 - alpha) * _zscore(all_scores) + alpha * _zscore(all_mahal)

    # Seen-class KNN bridge blending. Generic across ANY seen/unseen split —
    # bridge_alpha=0 disables it entirely (default), so this is a no-op unless
    # explicitly configured. As seen-class count shrinks (55/5 -> 30/30 ->
    # 12/48) the bridge signal gets noisier; tune/lower eval_bridge_alpha per
    # split rather than assuming one fixed value transfers across splits.
    bridge_alpha = float(config.get("eval_bridge_alpha", 0.0))
    if bridge_ctx is not None and bridge_alpha > 0.0 and bridge_score_chunks:
        all_bridge = torch.cat(bridge_score_chunks, dim=0)
        def _zscore_b(s: torch.Tensor) -> torch.Tensor:
            return (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(1e-6)
        all_scores = (1.0 - bridge_alpha) * _zscore_b(all_scores) + bridge_alpha * _zscore_b(all_bridge)

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
    if bool(config.get("eval_a3_calib", False)):
        all_scores = all_scores - all_scores.mean(dim=0, keepdim=True)

    if bool(config.get("eval_row_zscore", True)):
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

    pred_cols = torch.argmin(all_scores, dim=1)
    mapped = torch.tensor([label_to_idx[int(item)] for item in all_labels.detach().cpu()], device=device)
    confusion = torch.zeros((len(unseen_labels), len(unseen_labels)), device=device, dtype=torch.long)
    for true_idx, pred_idx in zip(mapped.view(-1), pred_cols.view(-1)):
        confusion[true_idx.long(), pred_idx.long()] += 1
    total = int(all_labels.numel())
    correct = int((pred_cols == mapped).sum().item()) if total else 0

    # Projection-space secondary accuracy (only when eval_also_proj_score).
    proj_acc: float = -1.0
    if eval_also_proj_score and proj_score_chunks:
        all_proj = torch.cat(proj_score_chunks, dim=0)
        if bool(config.get("eval_row_zscore", True)):
            all_proj = (all_proj - all_proj.mean(dim=1, keepdim=True)) / all_proj.std(dim=1, keepdim=True).clamp_min(1e-6)
        proj_pred = torch.argmin(all_proj, dim=1)
        proj_acc = float((proj_pred == mapped).sum().item()) / total if total else 0.0

    elapsed = time.time() - start_time
    return {
        "accuracy": correct / total if total else 0.0,
        "proj_accuracy": proj_acc,
        "cfg_scale": float(config.get("eval_cfg_scale", 1.0)),
        "use_mahal": use_mahal,
        "bridge_alpha": bridge_alpha,
        "num_samples": total,
        "elapsed_sec": elapsed,
        "samples_per_sec": total / elapsed if elapsed > 0 else 0.0,
        "unseen_labels": [int(item) for item in unseen_labels],
        "eval_timesteps": [int(item) for item in t_values],
        "eval_num_noise": int(noise_count),
        "eval_noise_seed": eval_seed,
        "eval_distance_space": "prediction" if use_native_eval_distance else "x0",
        "confusion_matrix": confusion.detach().cpu().numpy(),
        "true_labels": all_labels.detach().cpu().numpy(),
        "predicted_labels": np.asarray(
            [int(unseen_labels[int(index)]) for index in pred_cols.detach().cpu().numpy()],
            dtype=np.int64,
        ),
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
    elif model_type == "text_gate":
        ModelClass = DiTTextGate
    else:
        ModelClass = DiT
    model = ModelClass(
        **model_kwargs,
    ).to(device=device, dtype=dtype)

    loss_mode = str(config.get("loss_mode", "c2u_feat")).lower()
    use_c2u_loss = loss_mode in {"c2u_feat", "c2u_feat_infonce"}
    use_align_loss = loss_mode == "c2u_feat_align"
    aux = None
    use_projected_infonce = bool(config.get("projected_infonce_weight", 0.0)) and use_c2u_loss
    needs_aux = use_align_loss or (use_c2u_loss and (
        float(config.get("gen_con_weight", 0.0)) > 0.0
        or float(config.get("text_align_weight", 0.0)) > 0.0
        or use_projected_infonce
    ))
    if needs_aux:
        aux = C2UAuxHeads(
            hidden_size=int(config["hidden_size"]),
            text_size=int(text_embed.shape[-1]),
            feature_size=int(config.get("in_channels", 256)),
            proj_size=int(config.get("proj_size", 256)),
        ).to(device=device, dtype=dtype)

    trainable_params = list(model.parameters()) + (list(aux.parameters()) if aux is not None else [])
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    total_steps = int(config["num_iter"])
    lr_scheduler = get_scheduler(
        str(config.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(config.get("num_warmup", 0)),
        num_training_steps=total_steps,
    )

    global_step = 0
    start_epoch = 0
    best_acc = 0.0
    use_ema = bool(config.get("use_ema", False))
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if args.resume_from_checkpoint:
        metadata = load_checkpoint(
            repo_path(args.resume_from_checkpoint),
            model,
            optimizer,
            lr_scheduler,
            aux=aux,
            ema_model=ema_model,
        )
        global_step = int(metadata.get("global_step", 0))
        start_epoch = int(metadata.get("epoch", 0))
        best_acc = float(metadata.get("best_zsl_acc", 0.0))

    if bool(args.eval_only):
        # Precision matrix must be built before eval_only early-return.
        _eval_precision = None
        if bool(config.get("eval_mahal", False)):
            _shrink = float(config.get("eval_mahal_shrinkage", config.get("dcr_shrinkage", 0.1)))
            _eval_precision = estimate_shared_precision(config, unseen_labels, _shrink, device)
            print(f"Mahalanobis eval: precision matrix shape={tuple(_eval_precision.shape)}")
        _bridge_ctx = build_seen_bridge_context(config, unseen_labels, text_embed, device)
        if _bridge_ctx is not None:
            print(f"Seen-class bridge: {_bridge_ctx['x_seen'].shape[0]} seen train features, k={_bridge_ctx['k']}")
        _analogy_ctx = build_seen_analogy_context(config, unseen_labels, text_embed, device)
        eval_model = ema_model if ema_model is not None else model
        eval_aux = aux
        metrics = evaluate_zsl(eval_model, test_loader, scheduler, text_embed, unseen_labels, config, device, dtype, aux=eval_aux, precision=_eval_precision, bridge_ctx=_bridge_ctx, analogy_ctx=_analogy_ctx)
        print(
            json.dumps(
                {
                    "accuracy": metrics["accuracy"],
                    "num_samples": metrics["num_samples"],
                    "samples_per_sec": metrics["samples_per_sec"],
                    "eval_timesteps": metrics["eval_timesteps"],
                    "eval_num_noise": metrics["eval_num_noise"],
                    "eval_noise_seed": metrics["eval_noise_seed"],
                    "eval_distance_space": metrics["eval_distance_space"],
                    "use_ema": use_ema,
                    "checkpoint": args.resume_from_checkpoint,
                },
                indent=2,
            )
        )
        save_eval_artifacts(work_dir, metrics, args.eval_tag)
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
    total_epoch = int(config.get("num_epoch", 0)) or (total_steps // max(len(train_loader), 1) + 1)
    easy_bank, hard_bank = load_negative_bank(config, device)
    topk_hard_bank = load_topk_hard_bank(config, device)
    if topk_hard_bank is not None:
        k = topk_hard_bank.shape[1]
        print(f"TopK hard neg bank loaded: shape={tuple(topk_hard_bank.shape)}, K={k} (seen-only negatives)")
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
        print(f"Seen-class bridge: {eval_bridge_ctx['x_seen'].shape[0]} seen train features, k={eval_bridge_ctx['k']}")
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
                f"feature_tokens={int(config.get('feature_tokens', 1)) if ModelClass is DiTFineGrained else 1}"
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
                    f"distance_space={config.get('eval_distance_space', 'x0')}"
                ),
            )

    for epoch in range(start_epoch, total_epoch):
        model.train()
        if aux is not None:
            aux.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_easy_rank = 0.0
        epoch_hard_rank = 0.0
        epoch_infonce = 0.0
        epoch_proj_infonce = 0.0
        epoch_gen_con = 0.0
        epoch_text_align = 0.0
        epoch_dcr = 0.0
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
            timesteps = torch.randint(train_t_min, train_t_max + 1, (bsz,), device=device).long()
            noisy = scheduler.add_noise(features, noise, timesteps)
            target = diffusion_training_target(prediction_type, features, noise, timesteps, scheduler)
            fc, fl = split_text_condition(text_embed, labels)

            if loss_mode in {"c2u_feat", "c2u_feat_infonce"}:
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

                if loss_mode == "c2u_feat_infonce":
                    negative_labels = sample_infonce_negative_labels(
                        labels=labels,
                        seen_labels=seen_tensor_choices,
                        num_neg=int(config.get("infonce_num_neg", 7)),
                        device=device,
                        hard_bank=hard_bank,
                        topk_hard_bank=topk_hard_bank,
                        force_hard_pairs=force_hard_pairs,
                    )
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
            }:
                pred = model(noisy, timesteps.to(dtype), fc, fl)
                use_native_triplet = loss_mode in {
                    "tdsm_native_triplet",
                    "tdsm_prediction_triplet",
                }
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

                random_neg_labels = torch.tensor(
                    random.choices(seen_tensor_choices, k=bsz),
                    device=device,
                    dtype=torch.long,
                )
                random_fc, random_fl = split_text_condition(text_embed, random_neg_labels)
                pred_random = model(noisy, timesteps.to(dtype), random_fc, random_fl)
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
                random_valid = labels.ne(random_neg_labels)

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

                margin = float(config.get("tdsm_triplet_margin", config.get("margin_x0", config.get("margin", 0.1))))
                triplet_x0 = torch.clamp(pos_dist - neg_dist + margin, min=0.0) * mask

                recon = F.mse_loss(pred, target)
                easy_rank = torch.tensor(0.0, device=device)
                hard_rank = triplet_x0.mean()
                infonce = torch.tensor(0.0, device=device)
                projected_infonce = torch.tensor(0.0, device=device)
                gen_con = torch.tensor(0.0, device=device)
                text_align = torch.tensor(0.0, device=device)
                dcr = torch.tensor(0.0, device=device)
                loss = (
                    recon * float(config.get("rec_weight", config.get("d_weight", 1.0)))
                    + hard_rank * float(config.get("tdsm_triplet_weight", config.get("t_weight", 1.0)))
                )
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
            (loss / grad_accum_steps).backward()

            should_step = accum_count >= grad_accum_steps or (batch_idx + 1 == len(train_loader))
            if should_step:
                if accum_count != grad_accum_steps:
                    scale = grad_accum_steps / max(1, accum_count)
                    for param in trainable_params:
                        if param.grad is not None:
                            param.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(config.get("grad_clip", 1.0)))
                optimizer.step()
                lr_scheduler.step()
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
            epoch_batches += 1

            if global_step >= total_steps:
                break

        denom = max(1, epoch_batches)
        log(
            train_log,
            (
                f"Iter[{global_step}/{total_steps}]\tLoss: {epoch_loss / denom:.6f}\t"
                f"Recon: {epoch_recon / denom:.6f}\tEasyRank: {epoch_easy_rank / denom:.6f}\t"
                f"HardRank: {epoch_hard_rank / denom:.6f}\tInfoNCE: {epoch_infonce / denom:.6f}\t"
                f"ProjInfoNCE: {epoch_proj_infonce / denom:.6f}\t"
                f"GenCon: {epoch_gen_con / denom:.6f}\tTextAlign: {epoch_text_align / denom:.6f}\t"
                f"DCR: {epoch_dcr / denom:.6f}"
            ),
        )

        if eval_during_training and (epoch + 1) % int(config.get("eval_epoch", 1)) == 0:
            eval_model = ema_model if ema_model is not None else model
            eval_aux = aux
            metrics = evaluate_zsl(eval_model, test_loader, scheduler, text_embed, unseen_labels, config, device, dtype, aux=eval_aux, precision=eval_precision, bridge_ctx=eval_bridge_ctx, analogy_ctx=eval_analogy_ctx)
            log(
                val_log,
                (
                    f"ZSL {eval_split_name} Acc: {metrics['accuracy']:.6f}\tEpoch: {epoch + 1}\t"
                    f"Samples: {metrics['num_samples']}\tSpeed: {metrics['samples_per_sec']:.4f}\t"
                    f"EvalT: {metrics['eval_timesteps']}\tEvalNoise: {metrics['eval_num_noise']}\t"
                    f"EvalSeed: {metrics['eval_noise_seed']}\tEMA: {use_ema}\t"
                    f"EvalSpace: {metrics['eval_distance_space']}\t"
                    f"CFG: {metrics.get('cfg_scale', 1.0):.1f}\tMahal: {metrics.get('use_mahal', False)}"
                    + (f"\tProjAcc: {metrics['proj_accuracy']:.6f}" if metrics.get("proj_accuracy", -1) >= 0 else "")
                ),
            )
            save_eval_artifacts(work_dir, metrics, "latest")
            can_update_best = metrics["num_samples"] >= min_best_eval_samples
            if can_update_best and metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                save_eval_artifacts(work_dir, metrics, "best")
                save_checkpoint(
                    work_dir / "best",
                    model,
                    optimizer,
                    lr_scheduler,
                    {"epoch": epoch + 1, "global_step": global_step, "best_zsl_acc": best_acc},
                    aux=aux,
                    ema_model=ema_model,
                    save_ema_as_model=ema_model is not None,
                )
                log(
                    val_log,
                    f"Best {eval_split_name} Acc: {best_acc:.6f}\tBest Epoch: {epoch + 1}",
                )

        save_checkpoint(
            work_dir / "latest",
            model,
            optimizer,
            lr_scheduler,
            {"epoch": epoch + 1, "global_step": global_step, "best_zsl_acc": best_acc},
            aux=aux,
            ema_model=ema_model,
            save_ema_as_model=False,
        )
        if global_step >= total_steps:
            break

    train_log.close()
    val_log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train feature-space C2U x-prediction ZSL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--work-dir", default=None, help="Override config work_dir, useful for isolated smoke runs.")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-tag", default="eval_only")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.work_dir:
        config["work_dir"] = args.work_dir
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
