"""Mechanism-to-feature subspace suppression probe.

The direct blind-response probe showed that suppressing a parameter-gradient
direction is not enough: it skips the map from a mechanism direction to a
predictive feature direction.  This experiment implements that missing map:

    E = G P_ker(O) -> V_r -> M V_r -> U_spur.

Here ``G`` is the source environment gradient-response map, ``O`` is the
IRMv1 observation row, and ``M`` is a label-conditioned hidden-feature
association map.  The training penalty is the classifier loading in the
inferred feature subspace, ``||U_spur.T w||^2``.  The learner sees source
tensors and environment IDs only; target data are evaluation-only.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_hidden_env_finite_sample_tsr import (
    DEFAULT_EPOCHS,
    DEFAULT_IRM_LAMBDA,
    DEFAULT_LR,
    DEFAULT_REFRESH_EVERY,
    HIDDEN_DIM,
    INPUT_DIM,
    MLP,
    SOURCE_ENVS,
    baseline_row,
    contrast_matrix,
    evaluate,
    flatten_tensors,
    irm_statistics,
    make_datasets,
    make_geometry,
    null_projector,
    parameters_tuple,
    set_seed,
)


DEFAULT_WARMUP_EPOCHS = 6
DEFAULT_SUBSPACE_RANK = 2
DEFAULT_SUBSPACE_LAMBDA = 1.0
SUBSPACE_TOL = 1e-7


def environment_gradient_matrix(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    create_graph: bool,
) -> torch.Tensor:
    """Return the parameter-gradient matrix with one column per environment."""

    params = parameters_tuple(model)
    columns = []
    for x, y in envs:
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        gradients = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        columns.append(flatten_tensors(gradients))
    return torch.stack(columns, dim=1)


def label_association_matrix(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
) -> np.ndarray:
    """Estimate M=[m_e-m_bar] C_E, m_e=E[(2Y-1) z]."""

    associations = []
    with torch.no_grad():
        for x, y in envs:
            feature = model.feature(x)
            signed_label = 2.0 * y - 1.0
            associations.append(
                (feature * signed_label[:, None]).mean(dim=0).cpu().numpy()
            )
    association_matrix = np.stack(associations, axis=1)
    centered = association_matrix - association_matrix.mean(axis=1, keepdims=True)
    return centered @ contrasts


def orthogonal_feature_subspace(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    strategy: str,
    rank: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map raw or blind mechanism directions to a hidden-feature subspace."""

    hidden_dim = model.fc1.out_features
    if strategy == "full":
        return np.eye(hidden_dim), {
            "subspace_rank": hidden_dim,
            "selected_singular_values": [],
            "feature_direction_norms": [],
            "blind_beta": 0.0,
            "raw_response": 0.0,
        }

    gradients = environment_gradient_matrix(model, envs, False)
    contrast_tensor = torch.tensor(contrasts, dtype=gradients.dtype)
    response = (gradients @ contrast_tensor).detach().cpu().numpy()
    observation = baseline_row(model, envs, contrasts)
    projector = null_projector(observation[None, :])
    blind_response = response @ projector

    _, raw_singular_values, raw_right = np.linalg.svd(
        response, full_matrices=False
    )
    _, blind_singular_values, blind_right = np.linalg.svd(
        blind_response, full_matrices=False
    )
    if strategy == "raw":
        singular_values = raw_singular_values
        directions = raw_right[:rank].T
    elif strategy == "blind":
        singular_values = blind_singular_values
        directions = blind_right[:rank].T
    else:
        raise ValueError(strategy)

    label_map = label_association_matrix(model, envs, contrasts)
    selected_count = min(rank, directions.shape[1], len(singular_values))
    feature_directions = label_map @ directions[:, :selected_count]
    feature_direction_norms = np.linalg.norm(feature_directions, axis=0)
    if selected_count:
        weighted = feature_directions * np.sqrt(
            np.maximum(singular_values[:selected_count], 0.0)
        )[None, :]
        left, feature_singular_values, _ = np.linalg.svd(
            weighted, full_matrices=False
        )
        cutoff = SUBSPACE_TOL * max(
            float(feature_singular_values[0]) if len(feature_singular_values) else 0.0,
            1.0,
        )
        keep = int(np.sum(feature_singular_values > cutoff))
        feature_subspace = left[:, :keep]
    else:
        feature_subspace = np.zeros((hidden_dim, 0))

    diagnostics = {
        "subspace_rank": int(feature_subspace.shape[1]),
        "selected_singular_values": [
            float(value) for value in singular_values[:selected_count]
        ],
        "feature_direction_norms": [
            float(value) for value in feature_direction_norms
        ],
        "blind_beta": float(np.linalg.svd(blind_response, compute_uv=False)[0]),
        "raw_response": float(np.linalg.svd(response, compute_uv=False)[0]),
        "observation_norm": float(np.linalg.norm(observation)),
        "label_map_norm": float(np.linalg.norm(label_map)),
    }
    return feature_subspace, diagnostics


def feature_penalty(
    model: MLP,
    feature_subspace: torch.Tensor | None,
) -> torch.Tensor:
    if feature_subspace is None or feature_subspace.shape[1] == 0:
        return torch.zeros((), dtype=model.fc2.weight.dtype)
    loading = feature_subspace.T @ model.fc2.weight.squeeze(0)
    return loading.pow(2).sum()


def objective(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    feature_subspace: torch.Tensor | None,
    irm_lambda: float,
    subspace_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    irm_values = []
    for x, y in envs:
        logits = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logits, y))
        irm_values.append(irm_statistics(logits, y, create_graph=True))
    source_loss = torch.stack(losses).mean()
    irm_penalty = torch.stack(irm_values).pow(2).mean()
    suppression = feature_penalty(model, feature_subspace)
    total = source_loss + irm_lambda * irm_penalty + subspace_lambda * suppression
    return total, {
        "source_loss": float(source_loss.detach()),
        "irm_penalty": float(irm_penalty.detach()),
        "feature_suppression": float(suppression.detach()),
    }


def train_method(
    method: str,
    base_state: dict[str, torch.Tensor],
    source: list[tuple[torch.Tensor, torch.Tensor]],
    estimate_source: list[tuple[torch.Tensor, torch.Tensor]],
    target: tuple[torch.Tensor, torch.Tensor],
    contrasts: np.ndarray,
    seed: int,
    epochs: int,
    warmup_epochs: int,
    refresh_every: int,
    lr: float,
    irm_lambda: float,
    subspace_lambda: float,
    rank: int,
) -> dict:
    model = MLP()
    model.load_state_dict(copy.deepcopy(base_state))
    strategy = {
        "irm": "none",
        "raw_subspace": "raw",
        "blind_subspace": "blind",
        "full_feature": "full",
    }[method]
    if strategy == "none":
        subspace = None
        initial_diag = {
            "subspace_rank": 0,
            "selected_singular_values": [],
            "feature_direction_norms": [],
            "blind_beta": 0.0,
            "raw_response": 0.0,
        }
    else:
        subspace_np, initial_diag = orthogonal_feature_subspace(
            model, estimate_source, contrasts, strategy, rank
        )
        subspace = torch.tensor(subspace_np, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    refresh_count = 0
    turnover = 0
    steps = max(0, epochs - warmup_epochs)
    history = []
    subspace_history = [
        {
            "stage": "initial",
            "step": 0,
            "rank": initial_diag["subspace_rank"],
            "blind_beta": initial_diag["blind_beta"],
            "raw_response": initial_diag["raw_response"],
            "label_map_norm": initial_diag.get("label_map_norm", 0.0),
        }
    ]
    previous_subspace = subspace.detach().cpu().numpy() if subspace is not None else None

    for step in range(steps):
        if strategy in ("raw", "blind") and step % refresh_every == 0:
            new_np, refresh_diag = orthogonal_feature_subspace(
                model, estimate_source, contrasts, strategy, rank
            )
            if previous_subspace is not None and new_np.shape == previous_subspace.shape:
                if new_np.shape[1] == 0:
                    turnover += 0
                else:
                    alignment = np.linalg.svd(
                        previous_subspace.T @ new_np, compute_uv=False
                    )
                    turnover += int(float(alignment.mean()) < 0.99)
            subspace = torch.tensor(new_np, dtype=torch.float32)
            previous_subspace = new_np
            refresh_count += 1
            subspace_history.append(
                {
                    "stage": "refresh",
                    "step": step,
                    "rank": refresh_diag["subspace_rank"],
                    "blind_beta": refresh_diag["blind_beta"],
                    "raw_response": refresh_diag["raw_response"],
                    "label_map_norm": refresh_diag["label_map_norm"],
                }
            )

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(
            model,
            source,
            subspace,
            irm_lambda,
            subspace_lambda,
        )
        loss.backward()
        optimizer.step()
        history.append(metrics)

    source_x = torch.cat([x for x, _ in source])
    source_y = torch.cat([y for _, y in source])
    source_metric = evaluate(model, (source_x, source_y))
    target_metric = evaluate(model, target)
    final_subspace = subspace
    final_loading = float(feature_penalty(model, final_subspace).detach())
    return {
        "method": method,
        "source_loss": source_metric["loss"],
        "source_accuracy": source_metric["accuracy"],
        "target_loss": target_metric["loss"],
        "target_accuracy": target_metric["accuracy"],
        "initial_subspace_rank": initial_diag["subspace_rank"],
        "initial_blind_beta": initial_diag["blind_beta"],
        "initial_raw_response": initial_diag["raw_response"],
        "initial_feature_direction_norms": initial_diag["feature_direction_norms"],
        "final_feature_suppression": final_loading,
        "refresh_count": refresh_count,
        "subspace_turnover": turnover,
        "subspace_history": subspace_history,
        "epochs_after_warmup": steps,
        "history_last": history[-1] if history else {},
    }


def run_setting(
    n_per_env: int,
    gamma: float,
    seed: int,
    target_shift_mode: str,
    epochs: int,
    warmup_epochs: int,
    refresh_every: int,
    lr: float,
    irm_lambda: float,
    subspace_lambda: float,
    rank: int,
) -> dict:
    set_seed(seed)
    geometry = make_geometry(seed)
    source, estimate_source, target, target_a, target_offset = make_datasets(
        geometry, n_per_env, gamma, seed, target_shift_mode=target_shift_mode
    )
    contrasts = contrast_matrix(SOURCE_ENVS)
    warmup = MLP()
    warmup_optimizer = torch.optim.Adam(warmup.parameters(), lr=lr)
    for _ in range(warmup_epochs):
        warmup_optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(warmup, source, None, irm_lambda, 0.0)
        loss.backward()
        warmup_optimizer.step()
    base_state = copy.deepcopy(warmup.state_dict())

    methods = ["irm", "raw_subspace", "blind_subspace", "full_feature"]
    rows = [
        train_method(
            method,
            base_state,
            source,
            estimate_source,
            target,
            contrasts,
            seed,
            epochs,
            warmup_epochs,
            refresh_every,
            lr,
            irm_lambda,
            subspace_lambda,
            rank,
        )
        for method in methods
    ]
    return {
        "n_per_env": n_per_env,
        "gamma": gamma,
        "seed": seed,
        "target_shift_mode": target_shift_mode,
        "target_a_norm": float(np.linalg.norm(target_a)),
        "target_offset_norm": float(np.linalg.norm(target_offset)),
        "target_data_used_for_selection": False,
        "methods": methods,
        "rows": rows,
    }


def summarize(records: list[dict]) -> list[dict]:
    expanded = [
        row
        | {key: record[key] for key in ("n_per_env", "gamma", "target_shift_mode")}
        for record in records
        for row in record["rows"]
    ]
    grouped = itertools.groupby(
        sorted(
            expanded,
            key=lambda row: (
                row["target_shift_mode"],
                row["n_per_env"],
                row["gamma"],
                row["method"],
            ),
        ),
        key=lambda row: (
            row["target_shift_mode"],
            row["n_per_env"],
            row["gamma"],
            row["method"],
        ),
    )
    summary = []
    for key, group_iter in grouped:
        group = list(group_iter)
        item = {
            "target_shift_mode": key[0],
            "n_per_env": key[1],
            "gamma": key[2],
            "method": key[3],
        }
        for metric in (
            "target_loss",
            "target_accuracy",
            "initial_subspace_rank",
            "final_feature_suppression",
            "subspace_turnover",
        ):
            values = np.asarray([row[metric] for row in group], dtype=float)
            item[metric] = float(values.mean())
            item[metric + "_std"] = float(values.std())
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--target-shift-modes",
        nargs="+",
        choices=["predictive", "nuisance", "sign_reversal"],
        default=["predictive"],
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--refresh-every", type=int, default=DEFAULT_REFRESH_EVERY)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--irm-lambda", type=float, default=DEFAULT_IRM_LAMBDA)
    parser.add_argument("--subspace-lambda", type=float, default=DEFAULT_SUBSPACE_LAMBDA)
    parser.add_argument("--rank", type=int, default=DEFAULT_SUBSPACE_RANK)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for target_shift_mode in args.target_shift_modes:
        for n_per_env in args.sample_sizes:
            for gamma in args.gammas:
                for seed in args.seeds:
                    records.append(
                        run_setting(
                            n_per_env,
                            gamma,
                            seed,
                            target_shift_mode,
                            args.epochs,
                            args.warmup_epochs,
                            args.refresh_every,
                            args.lr,
                            args.irm_lambda,
                            args.subspace_lambda,
                            args.rank,
                        )
                    )

    output = {
        "configuration": {
            "source_envs": SOURCE_ENVS,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "rank": args.rank,
            "methods": ["irm", "raw_subspace", "blind_subspace", "full_feature"],
            "gradient_mode": "full_parameter_gradient",
            "feature_map": "label_conditioned_hidden_association",
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "refresh_every": args.refresh_every,
            "sample_sizes": args.sample_sizes,
            "gammas": args.gammas,
            "seeds": args.seeds,
            "target_shift_modes": args.target_shift_modes,
            "learning_rate": args.lr,
            "irm_lambda": args.irm_lambda,
            "subspace_lambda": args.subspace_lambda,
            "selection_uses_target": False,
            "selection_uses_latent_environment_coordinate": False,
        },
        "records": records,
        "summary": summarize(records),
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
