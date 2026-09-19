"""Blind-Spot Exposure (BSE) training probe.

This is a source-only algorithm experiment, separate from the oracle-based
blind-transfer attack validation.  It estimates an IRMv1 observation row and
the classifier-head transfer map in source-environment contrast coordinates,
then exposes one selected contrast through a Transfer-Measure-style
adversarial classifier-head task.

The selector is frozen between refreshes.  The target distribution is used
only for final evaluation.  The comparison is deliberately controlled:
IRMv1, Random-BSE, Raw-G-BSE, Blind-BSE, and Full-Transfer all start from the
same IRMv1 warm-up state and use the same inner classifier attack primitive.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_hidden_env_finite_sample_tsr import (
    DEFAULT_IRM_LAMBDA,
    DEFAULT_LR,
    HIDDEN_DIM,
    INPUT_DIM,
    MLP,
    SOURCE_ENVS,
    contrast_matrix,
    evaluate,
    irm_statistics,
    irm_statistics_nograd,
    make_datasets,
    make_geometry,
    parameters_tuple,
    set_seed,
)


torch.set_num_threads(1)

DEFAULT_EPOCHS = 24
DEFAULT_WARMUP_EPOCHS = 6
DEFAULT_REFRESH_EVERY = 5
DEFAULT_ATTACK_STEPS = 5
DEFAULT_ATTACK_LR = 0.08
DEFAULT_ATTACK_RADIUS = 0.35
DEFAULT_TRANSFER_LAMBDA = 2.0
DEFAULT_ALPHA = 0.8


def null_projector(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=float).reshape(1, -1)
    if row.size == 0:
        return np.eye(row.shape[1])
    projection = np.linalg.pinv(row, rcond=1e-8) @ row
    return np.eye(row.shape[1]) - (projection + projection.T) / 2.0


def head_vector(model: MLP) -> torch.Tensor:
    return torch.cat([model.fc2.weight.reshape(-1), model.fc2.bias.reshape(-1)])


def head_risk(
    features: torch.Tensor, labels: torch.Tensor, head: torch.Tensor
) -> torch.Tensor:
    logits = features @ head[:-1] + head[-1]
    return F.binary_cross_entropy_with_logits(logits, labels)


def head_transfer_matrix(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
) -> np.ndarray:
    """Full classifier-head gradient/Hessian geometry, one column per env."""

    columns = []
    for x, y in envs:
        features = model.feature(x).detach()
        head = head_vector(model).detach().requires_grad_(True)

        def risk(vector: torch.Tensor) -> torch.Tensor:
            return head_risk(features, y, vector)

        gradient = torch.autograd.grad(risk(head), head, create_graph=False)[0]
        hessian = torch.autograd.functional.hessian(risk, head)
        columns.append(
            torch.cat([gradient.detach(), 0.5 * hessian.reshape(-1).detach()])
            .cpu()
            .numpy()
        )
    return np.stack(columns, axis=1)


def irm_observation(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
) -> np.ndarray:
    values = [irm_statistics_nograd(model(x), y) for x, y in envs]
    row = np.asarray(values, dtype=float) @ contrasts
    return row / max(np.linalg.norm(row), 1e-8)


def geometry(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict[str, float]]:
    observation = irm_observation(model, envs, contrasts)
    transfer = head_transfer_matrix(model, envs) @ contrasts
    projector = null_projector(observation)
    blind_matrix = transfer @ projector
    _, singular_values, right = np.linalg.svd(blind_matrix, full_matrices=False)
    raw_singular_values, raw_right = np.linalg.svd(transfer, full_matrices=False)[1:]
    if strategy == "blind":
        vector = right[0]
    elif strategy == "raw":
        vector = raw_right[0]
    elif strategy == "random":
        vector = rng.normal(size=transfer.shape[1])
    elif strategy == "full":
        return [np.eye(transfer.shape[1])[:, j] for j in range(transfer.shape[1])], {
            "beta": float(singular_values[0]),
            "raw_response": float(raw_singular_values[0]),
            "observation_norm": float(np.linalg.norm(observation)),
            "selected_response": float(np.linalg.norm(transfer)),
            "selected_observation": 0.0,
        }
    elif strategy == "irm":
        return [], {
            "beta": float(singular_values[0]),
            "raw_response": float(raw_singular_values[0]),
            "observation_norm": float(np.linalg.norm(observation)),
            "selected_response": 0.0,
            "selected_observation": 0.0,
        }
    else:
        raise ValueError(strategy)
    vector = np.asarray(vector, dtype=float)
    vector /= max(np.linalg.norm(vector), 1e-12)
    return [vector], {
        "beta": float(singular_values[0]),
        "raw_response": float(raw_singular_values[0]),
        "observation_norm": float(np.linalg.norm(observation)),
        "selected_response": float(np.linalg.norm(transfer @ vector)),
        "selected_observation": float(abs(observation @ vector)),
    }


def contrast_pair(
    direction: np.ndarray,
    contrasts: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    env_direction = contrasts @ direction
    weights = np.ones(SOURCE_ENVS) / SOURCE_ENVS
    tau_max = min(
        float(weights[index] / abs(value))
        for index, value in enumerate(env_direction)
        if abs(value) > 1e-10
    )
    tau = alpha * tau_max
    return weights + tau * env_direction, weights - tau * env_direction, tau


def weighted_risk(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    weights: np.ndarray,
    head: torch.Tensor | None = None,
) -> torch.Tensor:
    values = []
    for weight, (x, y) in zip(weights, envs):
        features = model.feature(x)
        current_head = head if head is not None else head_vector(model)
        values.append(float(weight) * head_risk(features, y, current_head))
    return torch.stack(values).sum()


def pair_gap(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    weights_plus: np.ndarray,
    weights_minus: np.ndarray,
    head: torch.Tensor,
) -> torch.Tensor:
    return weighted_risk(model, envs, weights_plus, head) - weighted_risk(
        model, envs, weights_minus, head
    )


def adversarial_head(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    weights_plus: np.ndarray,
    weights_minus: np.ndarray,
    radius: float,
    steps: int,
    attack_lr: float,
) -> tuple[torch.Tensor, float]:
    features = [model.feature(x).detach() for x, _ in envs]
    labels = [y for _, y in envs]
    start = head_vector(model).detach()
    delta = torch.zeros_like(start, requires_grad=True)

    def detached_gap(vector: torch.Tensor) -> torch.Tensor:
        plus = sum(
            float(weight) * head_risk(feature, label, vector)
            for weight, feature, label in zip(weights_plus, features, labels)
        )
        minus = sum(
            float(weight) * head_risk(feature, label, vector)
            for weight, feature, label in zip(weights_minus, features, labels)
        )
        return plus - minus

    for _ in range(steps):
        objective = detached_gap(start + delta).pow(2)
        gradient = torch.autograd.grad(objective, delta)[0]
        delta = (delta + attack_lr * gradient / gradient.norm().clamp_min(1e-12)).detach()
        delta = (
            delta * min(1.0, radius / delta.norm().clamp_min(1e-12))
        ).requires_grad_(True)
    head = (start + delta.detach()).detach()
    return head, float(abs(detached_gap(head).detach()))


def base_objective(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    irm_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    irm_values = []
    for x, y in envs:
        logits = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logits, y))
        irm_values.append(irm_statistics(logits, y, create_graph=True))
    source_loss = torch.stack(losses).mean()
    irm_penalty = torch.stack(irm_values).pow(2).mean()
    return source_loss + irm_lambda * irm_penalty, {
        "source_loss": float(source_loss.detach()),
        "irm_penalty": float(irm_penalty.detach()),
    }


def train_method(
    method: str,
    base_state: dict[str, torch.Tensor],
    source: list[tuple[torch.Tensor, torch.Tensor]],
    estimate_source: list[tuple[torch.Tensor, torch.Tensor]],
    target: tuple[torch.Tensor, torch.Tensor],
    contrasts: np.ndarray,
    epochs: int,
    warmup_epochs: int,
    refresh_every: int,
    lr: float,
    irm_lambda: float,
    transfer_lambda: float,
    attack_radius: float,
    attack_steps: int,
    attack_lr: float,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    model = MLP()
    model.load_state_dict(copy.deepcopy(base_state))
    method_offset = {
        "irm": 0,
        "random-bse": 1,
        "raw-g-bse": 2,
        "blind-bse": 3,
        "full-transfer": 4,
    }[method]
    rng = np.random.default_rng(seed + 7000 + method_offset)
    strategy = {"irm": "irm", "random-bse": "random", "raw-g-bse": "raw", "blind-bse": "blind", "full-transfer": "full"}[method]
    directions, initial_diag = geometry(model, estimate_source, contrasts, strategy, rng)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    refresh_count = 0
    history = []
    pair_attack_history = []
    for step in range(max(0, epochs - warmup_epochs)):
        if method not in ("irm", "full-transfer") and step % refresh_every == 0:
            directions, refresh_diag = geometry(model, estimate_source, contrasts, strategy, rng)
            refresh_count += 1
        pairs = [contrast_pair(direction, contrasts, alpha) for direction in directions]
        adversarial_heads = []
        source_attack_gaps = []
        for weights_plus, weights_minus, _ in pairs:
            head, gap = adversarial_head(
                model,
                source,
                weights_plus,
                weights_minus,
                attack_radius,
                attack_steps,
                attack_lr,
            )
            adversarial_heads.append(head)
            source_attack_gaps.append(gap)

        optimizer.zero_grad(set_to_none=True)
        objective, metrics = base_objective(model, source, irm_lambda)
        transfer_penalty = torch.zeros_like(objective)
        for (weights_plus, weights_minus, _), head in zip(pairs, adversarial_heads):
            gap = pair_gap(model, source, weights_plus, weights_minus, head)
            transfer_penalty = transfer_penalty + gap.pow(2)
        if pairs:
            transfer_penalty = transfer_penalty / len(pairs)
            objective = objective + transfer_lambda * transfer_penalty
        objective.backward()
        optimizer.step()
        metrics["transfer_penalty"] = float(transfer_penalty.detach())
        metrics["source_attack_gap"] = float(np.mean(source_attack_gaps)) if source_attack_gaps else 0.0
        history.append(metrics)
        pair_attack_history.append(float(np.mean(source_attack_gaps)) if source_attack_gaps else 0.0)

    _, final_diag = geometry(model, estimate_source, contrasts, strategy, rng)
    source_x = torch.cat([x for x, _ in source])
    source_y = torch.cat([y for _, y in source])
    source_metric = evaluate(model, (source_x, source_y))
    target_metric = evaluate(model, target)
    return {
        "method": method,
        "source_loss": source_metric["loss"],
        "source_accuracy": source_metric["accuracy"],
        "target_loss": target_metric["loss"],
        "target_accuracy": target_metric["accuracy"],
        "initial_beta": initial_diag["beta"],
        "final_beta": final_diag["beta"],
        "initial_selected_response": initial_diag["selected_response"],
        "final_selected_response": final_diag["selected_response"],
        "initial_selected_observation": initial_diag["selected_observation"],
        "refresh_count": refresh_count,
        "mean_source_attack_gap_before": float(np.mean(pair_attack_history[:1])) if pair_attack_history else 0.0,
        "mean_source_attack_gap_after": float(np.mean(pair_attack_history[-1:])) if pair_attack_history else 0.0,
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
    transfer_lambda: float,
    attack_radius: float,
    attack_steps: int,
    attack_lr: float,
    alpha: float,
) -> dict[str, object]:
    set_seed(seed)
    geometry_data = make_geometry(seed)
    source, estimate_source, target, target_a, target_offset = make_datasets(
        geometry_data,
        n_per_env,
        gamma,
        seed,
        target_shift_mode=target_shift_mode,
    )
    contrasts = contrast_matrix(SOURCE_ENVS)
    warmup = MLP()
    warmup_optimizer = torch.optim.Adam(warmup.parameters(), lr=lr)
    for _ in range(warmup_epochs):
        warmup_optimizer.zero_grad(set_to_none=True)
        objective, _ = base_objective(warmup, source, irm_lambda)
        objective.backward()
        warmup_optimizer.step()
    base_state = copy.deepcopy(warmup.state_dict())
    methods = ["irm", "random-bse", "raw-g-bse", "blind-bse", "full-transfer"]
    rows = [
        train_method(
            method,
            base_state,
            source,
            estimate_source,
            target,
            contrasts,
            epochs,
            warmup_epochs,
            refresh_every,
            lr,
            irm_lambda,
            transfer_lambda,
            attack_radius,
            attack_steps,
            attack_lr,
            alpha,
            seed,
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
        "latent_coordinates_used_for_selection": False,
        "rows": rows,
    }


def summarize(records: list[dict[str, object]]) -> list[dict[str, float | str]]:
    rows = [
        row | {"target_shift_mode": record["target_shift_mode"]}
        for record in records
        for row in record["rows"]
    ]
    methods = sorted({row["method"] for row in rows})
    summary = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        item: dict[str, float | str] = {"method": method}
        for metric in (
            "target_loss",
            "target_accuracy",
            "initial_beta",
            "final_beta",
            "mean_source_attack_gap_before",
            "mean_source_attack_gap_after",
        ):
            values = np.asarray([row[metric] for row in group], dtype=float)
            item[metric] = float(values.mean())
            item[metric + "_std"] = float(values.std())
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target-shift-mode", choices=["predictive", "nuisance", "sign_reversal"], default="predictive")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--refresh-every", type=int, default=DEFAULT_REFRESH_EVERY)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--irm-lambda", type=float, default=DEFAULT_IRM_LAMBDA)
    parser.add_argument("--transfer-lambda", type=float, default=DEFAULT_TRANSFER_LAMBDA)
    parser.add_argument("--attack-radius", type=float, default=DEFAULT_ATTACK_RADIUS)
    parser.add_argument("--attack-steps", type=int, default=DEFAULT_ATTACK_STEPS)
    parser.add_argument("--attack-lr", type=float, default=DEFAULT_ATTACK_LR)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    records = [
        run_setting(
            args.sample_size,
            args.gamma,
            seed,
            args.target_shift_mode,
            args.epochs,
            args.warmup_epochs,
            args.refresh_every,
            args.lr,
            args.irm_lambda,
            args.transfer_lambda,
            args.attack_radius,
            args.attack_steps,
            args.attack_lr,
            args.alpha,
        )
        for seed in args.seeds
    ]
    output = {
        "configuration": {
            "source_envs": SOURCE_ENVS,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "methods": ["irm", "random-bse", "raw-g-bse", "blind-bse", "full-transfer"],
            "sample_size": args.sample_size,
            "gamma": args.gamma,
            "seeds": args.seeds,
            "target_shift_mode": args.target_shift_mode,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "refresh_every": args.refresh_every,
            "irm_lambda": args.irm_lambda,
            "transfer_lambda": args.transfer_lambda,
            "attack_radius": args.attack_radius,
            "attack_steps": args.attack_steps,
            "alpha": args.alpha,
            "selection_uses_target": False,
            "selection_uses_latent_environment_coordinate": False,
            "selector_stop_gradient": True,
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
