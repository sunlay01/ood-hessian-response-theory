"""Finite-sample probe for direct blind-response regularization.

This experiment removes the candidate-statistic actuation step from TSR.  At a
warm-start or refresh state, it estimates the source gradient-response map

    G = [g_1, ..., g_E] C_E

and the IRMv1 observation row O in the same source contrast coordinates.  The
blind method chooses the top right singular vector of ``G P_ker(O)`` and, for
the next inner block, directly penalizes ``||G v||^2``.  The raw control uses
the top right singular vector of ``G`` without the blind projection.

All selection uses source environments only.  The target is evaluation-only.
The implementation is gradient-only and uses full parameter gradients rather
than random sketches so that the ablation is an exact implementation of the
proposed beta objective on this small MLP.
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
    RANK_BUDGET,
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


DEFAULT_RESPONSE_LAMBDA = 0.5
DEFAULT_WARMUP_EPOCHS = 6


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


def response_geometry(
    model: MLP,
    estimate_envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Estimate a fixed contrast direction and its beta/raw diagnostics."""

    gradients = environment_gradient_matrix(model, estimate_envs, False)
    contrast_tensor = torch.tensor(contrasts, dtype=gradients.dtype)
    g = (gradients @ contrast_tensor).detach().cpu().numpy()
    observation = baseline_row(model, estimate_envs, contrasts)
    projector = null_projector(observation[None, :])
    blind = g @ projector

    if strategy == "blind":
        matrix = projector @ g.T @ g @ projector
        eigenvalues, eigenvectors = np.linalg.eigh(
            (matrix + matrix.T) / 2.0
        )
        vector = projector @ eigenvectors[:, -1]
    elif strategy == "raw":
        _, _, right = np.linalg.svd(g, full_matrices=False)
        vector = right[0]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=g.shape[1])
    elif strategy == "none":
        vector = None
    else:
        raise ValueError(strategy)

    if vector is not None:
        vector = np.asarray(vector, dtype=float)
        vector /= max(np.linalg.norm(vector), 1e-12)
    beta = float(np.linalg.svd(blind, compute_uv=False)[0])
    raw = float(np.linalg.svd(g, compute_uv=False)[0])
    diagnostics = {
        "beta": beta,
        "raw_response": raw,
        "observation_norm": float(np.linalg.norm(observation)),
        "blind_rank": float(np.linalg.matrix_rank(blind, tol=1e-8)),
        "vector_observation_abs": (
            float(abs(observation @ vector)) if vector is not None else 0.0
        ),
        "vector_response": (
            float(np.linalg.norm(g @ vector)) if vector is not None else 0.0
        ),
    }
    return vector, diagnostics


def objective(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    method: str,
    vector: torch.Tensor | None,
    irm_lambda: float,
    response_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    irm_values = []
    for x, y in envs:
        logits = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logits, y))
        irm_values.append(irm_statistics(logits, y, create_graph=True))
    source_loss = torch.stack(losses).mean()
    irm_penalty = torch.stack(irm_values).pow(2).mean()

    response_penalty = torch.zeros_like(source_loss)
    if method != "irm":
        gradients = environment_gradient_matrix(model, envs, True)
        contrast_tensor = torch.tensor(contrasts, dtype=gradients.dtype)
        response = gradients @ contrast_tensor
        if method == "full":
            # Average over contrast directions so the scale is comparable to
            # a unit-vector response penalty.
            response_penalty = response.pow(2).sum() / response.shape[1]
        else:
            if vector is None:
                raise ValueError(f"missing contrast vector for {method}")
            response_penalty = response @ vector
            response_penalty = response_penalty.pow(2).sum()

    total = source_loss + irm_lambda * irm_penalty + response_lambda * response_penalty
    return total, {
        "source_loss": float(source_loss.detach()),
        "irm_penalty": float(irm_penalty.detach()),
        "response_penalty": float(response_penalty.detach()),
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
    response_lambda: float,
) -> dict:
    model = MLP()
    model.load_state_dict(copy.deepcopy(base_state))
    vector_np, initial_diag = response_geometry(
        model,
        estimate_source,
        contrasts,
        {"blind": "blind", "raw": "raw", "random": "random", "full": "none", "irm": "none"}[method],
        seed + 4100,
    )
    vector = (
        torch.tensor(vector_np, dtype=torch.float32)
        if vector_np is not None
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    refresh_count = 0
    turnover = 0
    last_vector = vector_np.copy() if vector_np is not None else None
    history = []
    steps = max(0, epochs - warmup_epochs)
    for step in range(steps):
        if method in ("blind", "raw") and step % refresh_every == 0:
            new_vector, _ = response_geometry(
                model,
                estimate_source,
                contrasts,
                method,
                seed + 5000 + step,
            )
            if last_vector is not None and new_vector is not None:
                # Sign is immaterial to the squared response; use absolute
                # alignment to measure genuine eigendirection turnover.
                turnover += int(abs(float(last_vector @ new_vector)) < 0.99)
            vector_np = new_vector
            vector = torch.tensor(vector_np, dtype=torch.float32)
            last_vector = vector_np.copy()
            refresh_count += 1

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(
            model,
            source,
            contrasts,
            method,
            vector,
            irm_lambda,
            response_lambda,
        )
        loss.backward()
        optimizer.step()
        history.append(metrics)

    source_x = torch.cat([x for x, _ in source])
    source_y = torch.cat([y for _, y in source])
    source_metric = evaluate(model, (source_x, source_y))
    target_metric = evaluate(model, target)
    _, final_diag = response_geometry(
        model,
        estimate_source,
        contrasts,
        {"blind": "blind", "raw": "raw", "random": "random", "full": "none", "irm": "none"}[method],
        seed + 9000,
    )
    return {
        "method": method,
        "source_loss": source_metric["loss"],
        "source_accuracy": source_metric["accuracy"],
        "target_loss": target_metric["loss"],
        "target_accuracy": target_metric["accuracy"],
        "initial_beta": initial_diag["beta"],
        "initial_raw_response": initial_diag["raw_response"],
        "initial_vector_response": initial_diag["vector_response"],
        "initial_vector_observation_abs": initial_diag["vector_observation_abs"],
        "final_beta": final_diag["beta"],
        "final_raw_response": final_diag["raw_response"],
        "final_vector_response": final_diag["vector_response"],
        "refresh_count": refresh_count,
        "vector_turnover": turnover,
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
    response_lambda: float,
) -> dict:
    set_seed(seed)
    geometry = make_geometry(seed)
    source, estimate_source, target, target_a, target_offset = make_datasets(
        geometry,
        n_per_env,
        gamma,
        seed,
        target_shift_mode=target_shift_mode,
    )
    contrasts = contrast_matrix(SOURCE_ENVS)
    warmup = MLP()
    # All methods start from the same IRMv1 warmup state.
    warmup_optimizer = torch.optim.Adam(warmup.parameters(), lr=lr)
    for _ in range(warmup_epochs):
        warmup_optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(
            warmup,
            source,
            contrasts,
            "irm",
            None,
            irm_lambda,
            0.0,
        )
        loss.backward()
        warmup_optimizer.step()
    base_state = copy.deepcopy(warmup.state_dict())

    methods = ["irm", "random", "full", "blind", "raw"]
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
            response_lambda,
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
        "rows": rows,
    }


def summarize(records: list[dict]) -> list[dict]:
    expanded = [
        row
        | {
            key: record[key]
            for key in ("n_per_env", "gamma", "target_shift_mode")
        }
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
            "initial_beta",
            "final_beta",
            "initial_raw_response",
            "final_raw_response",
            "vector_turnover",
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
    parser.add_argument("--response-lambda", type=float, default=DEFAULT_RESPONSE_LAMBDA)
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
                            args.response_lambda,
                        )
                    )

    output = {
        "configuration": {
            "source_envs": SOURCE_ENVS,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "rank_budget": RANK_BUDGET,
            "methods": ["irm", "random", "full", "blind", "raw"],
            "gradient_mode": "full_parameter_gradient",
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "refresh_every": args.refresh_every,
            "sample_sizes": args.sample_sizes,
            "gammas": args.gammas,
            "seeds": args.seeds,
            "target_shift_modes": args.target_shift_modes,
            "learning_rate": args.lr,
            "irm_lambda": args.irm_lambda,
            "response_lambda": args.response_lambda,
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
