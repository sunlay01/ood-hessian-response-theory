"""Direct validation of the signed IRMv1 actuation-response prediction.

Beta is not an action criterion.  This experiment instead tests the local
algorithm theorem on a frozen classifier head:

    v_IRM = H_source^{-1} grad Omega_IRM,
    q(h)  = - <G_g h, v_IRM>,
    Delta_reg(h) - Delta_reg(0) ~= lambda * epsilon * q(h).

All directions live in the source-environment contrast space.  No target data
or latent environment coordinates are used.  The experiment also checks the
IRM residual norm at a separately optimized IRMv1 head; if it is near zero,
the first-order actuation interpretation must be replaced by constrained
manifold selection.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_hidden_env_finite_sample_tsr import (
    DEFAULT_IRM_LAMBDA,
    DEFAULT_LR,
    MLP,
    SOURCE_ENVS,
    contrast_matrix,
    irm_statistics,
    make_datasets,
    make_geometry,
    set_seed,
)


torch.set_num_threads(1)

DEFAULT_FEATURE_EPOCHS = 40
DEFAULT_HEAD_STEPS = 500
DEFAULT_RANDOM_DIRECTIONS = 128


def train_feature_model(
    source: list[tuple[torch.Tensor, torch.Tensor]], epochs: int, lr: float
) -> MLP:
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = [
            F.binary_cross_entropy_with_logits(model(x), y) for x, y in source
        ]
        torch.stack(losses).mean().backward()
        optimizer.step()
    return model


def frozen_environment_data(
    model: MLP, source: list[tuple[torch.Tensor, torch.Tensor]]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(model.feature(x).detach(), y) for x, y in source]


def head_risk(
    head: torch.Tensor, data: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    feature, label = data
    logits = feature @ head[:-1] + head[-1]
    return F.binary_cross_entropy_with_logits(logits, label)


def irm_residual(
    head: torch.Tensor, data: tuple[torch.Tensor, torch.Tensor], create_graph: bool
) -> torch.Tensor:
    feature, label = data
    logits = feature @ head[:-1] + head[-1]
    return irm_statistics(logits, label, create_graph=create_graph)


def head_objective(
    head: torch.Tensor,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    irm_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    risks = torch.stack([head_risk(head, env) for env in envs])
    residuals = torch.stack(
        [irm_residual(head, env, create_graph=True) for env in envs]
    )
    penalty = residuals.pow(2).mean()
    return risks.mean() + irm_lambda * penalty, risks.mean(), residuals


def optimize_head(
    start: torch.Tensor,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    irm_lambda: float,
    steps: int,
) -> torch.Tensor:
    head = start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([head], lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = head_objective(head, envs, irm_lambda)
        objective.backward()
        optimizer.step()
    return head.detach()


def source_geometry(
    head: torch.Tensor,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    damping: float,
) -> dict[str, np.ndarray | float]:
    variable = head.detach().clone().requires_grad_(True)

    def mean_risk(value: torch.Tensor) -> torch.Tensor:
        return torch.stack([head_risk(value, env) for env in envs]).mean()

    residuals = torch.stack(
        [irm_residual(variable, env, create_graph=True) for env in envs]
    )
    omega = residuals.pow(2).mean()
    grad_omega = torch.autograd.grad(omega, variable, create_graph=False)[0]
    hessian = torch.autograd.functional.hessian(mean_risk, variable)
    identity = torch.eye(hessian.shape[0], dtype=hessian.dtype)
    actuation = torch.linalg.solve(hessian + damping * identity, grad_omega)

    gradients = []
    for env in envs:
        current = head.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(head_risk(current, env), current)[0]
        gradients.append(gradient.detach().cpu().numpy())
    gradient_matrix = np.stack(gradients, axis=1)
    g_map = gradient_matrix @ contrasts
    observation = residuals.detach().cpu().numpy() @ contrasts
    observation = observation.reshape(1, -1)
    projector = np.eye(observation.shape[1]) - np.linalg.pinv(
        observation, rcond=1e-8
    ) @ observation
    projector = (projector + projector.T) / 2.0
    return {
        "g_map": g_map,
        "observation": observation,
        "projector": projector,
        "actuation": actuation.detach().cpu().numpy(),
        "mean_gradient_norm": float(
            np.linalg.norm(gradient_matrix.mean(axis=1))
        ),
        "grad_omega_norm": float(grad_omega.norm().detach()),
        "actuation_norm": float(actuation.norm().detach()),
        "residual_rms": float(torch.sqrt(residuals.pow(2).mean()).detach()),
        "omega": float(omega.detach()),
    }


def mixture_risk(
    head: np.ndarray,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    weights: np.ndarray,
) -> float:
    vector = torch.tensor(head, dtype=envs[0][0].dtype)
    risks = np.asarray(
        [float(head_risk(vector, env).detach()) for env in envs], dtype=float
    )
    return float(weights @ risks)


def oriented_svd_vector(matrix: np.ndarray) -> np.ndarray:
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    vector = right[0]
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0:
        vector = -vector
    return vector / max(np.linalg.norm(vector), 1e-12)


def prediction_and_harm(
    direction: np.ndarray,
    g_map: np.ndarray,
    actuation: np.ndarray,
    contrasts: np.ndarray,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    head: np.ndarray,
    epsilon: float,
    lambda_reg: float,
) -> tuple[float, float, float]:
    q_value = -float(actuation @ (g_map @ direction))
    base_weights = np.ones(SOURCE_ENVS) / SOURCE_ENVS
    shifted_weights = base_weights + epsilon * (contrasts @ direction)
    if np.min(shifted_weights) < -1e-10:
        raise ValueError("epsilon leaves the source-mixture simplex")
    stepped = head - lambda_reg * actuation
    base_change = mixture_risk(stepped, envs, base_weights) - mixture_risk(
        head, envs, base_weights
    )
    shifted_change = mixture_risk(stepped, envs, shifted_weights) - mixture_risk(
        head, envs, shifted_weights
    )
    centered_harm = shifted_change - base_change
    prediction = lambda_reg * epsilon * q_value
    return q_value, prediction, centered_harm


def correlation(x: list[float], y: list[float]) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_seed(
    seed: int,
    n_per_env: int,
    feature_epochs: int,
    head_steps: int,
    irm_lambda: float,
    epsilon: float,
    lambda_reg: float,
    damping: float,
    random_directions: int,
) -> dict[str, object]:
    set_seed(seed)
    data_geometry = make_geometry(seed)
    source, _, _, _, _ = make_datasets(
        data_geometry, n_per_env, gamma=1.0, seed=seed
    )
    contrasts = contrast_matrix(SOURCE_ENVS)
    model = train_feature_model(source, feature_epochs, DEFAULT_LR)
    envs = frozen_environment_data(model, source)
    start = torch.cat(
        [model.fc2.weight.detach().reshape(-1), model.fc2.bias.detach().reshape(-1)]
    )
    erm_head = optimize_head(start, envs, irm_lambda=0.0, steps=head_steps)
    irm_head = optimize_head(
        erm_head, envs, irm_lambda=irm_lambda, steps=head_steps
    )
    geometry_erm = source_geometry(
        erm_head, envs, contrasts, damping=damping
    )
    geometry_irm = source_geometry(
        irm_head, envs, contrasts, damping=damping
    )
    g_map = np.asarray(geometry_erm["g_map"])
    projector = np.asarray(geometry_erm["projector"])
    actuation = np.asarray(geometry_erm["actuation"])
    blind_map = g_map @ projector

    h_beta = oriented_svd_vector(blind_map)
    harmful_covector = projector @ g_map.T @ actuation
    chi = float(np.linalg.norm(harmful_covector))
    h_harm = -harmful_covector / max(chi, 1e-12)
    rng = np.random.default_rng(seed + 9001)
    random_blind = []
    for _ in range(random_directions):
        direction = projector @ rng.normal(size=projector.shape[0])
        direction /= max(np.linalg.norm(direction), 1e-12)
        random_blind.append(direction)

    direction_rows = []
    for name, direction in [("beta", h_beta), ("harm", h_harm)]:
        q_value, prediction, harm = prediction_and_harm(
            direction,
            g_map,
            actuation,
            contrasts,
            envs,
            erm_head.cpu().numpy(),
            epsilon,
            lambda_reg,
        )
        direction_rows.append(
            {
                "selector": name,
                "q": q_value,
                "predicted_harm": prediction,
                "actual_centered_harm": harm,
                "response_norm": float(np.linalg.norm(g_map @ direction)),
                "observation_abs": float(np.linalg.norm(np.asarray(geometry_erm["observation"]) @ direction)),
            }
        )

    random_rows = []
    for direction in random_blind:
        q_value, prediction, harm = prediction_and_harm(
            direction,
            g_map,
            actuation,
            contrasts,
            envs,
            erm_head.cpu().numpy(),
            epsilon,
            lambda_reg,
        )
        random_rows.append(
            {
                "q": q_value,
                "predicted_harm": prediction,
                "actual_centered_harm": harm,
                "response_norm": float(np.linalg.norm(g_map @ direction)),
            }
        )
    return {
        "seed": seed,
        "erm_geometry": {
            key: value
            for key, value in geometry_erm.items()
            if isinstance(value, float)
        },
        "irm_solution_geometry": {
            key: value
            for key, value in geometry_irm.items()
            if isinstance(value, float)
        },
        "regime": (
            "zero_residual" if float(geometry_irm["residual_rms"]) < 1e-3 else "nonzero_residual"
        ),
        "chi": chi,
        "beta": float(np.linalg.svd(blind_map, compute_uv=False)[0]),
        "direction_rows": direction_rows,
        "random_blind": random_rows,
        "random_q_harm_corr": correlation(
            [row["q"] for row in random_rows],
            [row["actual_centered_harm"] for row in random_rows],
        ),
        "random_response_harm_corr": correlation(
            [row["response_norm"] for row in random_rows],
            [row["actual_centered_harm"] for row in random_rows],
        ),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "nonzero_residual_seeds": sum(
            record["regime"] == "nonzero_residual" for record in records
        )
    }
    for metric in (
        "random_q_harm_corr",
        "random_response_harm_corr",
        "chi",
        "beta",
    ):
        values = np.asarray([record[metric] for record in records], dtype=float)
        result[metric + "_mean"] = float(values.mean())
        result[metric + "_std"] = float(values.std())
    for selector in ("beta", "harm"):
        rows = [
            next(row for row in record["direction_rows"] if row["selector"] == selector)
            for record in records
        ]
        for metric in ("q", "predicted_harm", "actual_centered_harm", "response_norm"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            result[f"{selector}_{metric}_mean"] = float(values.mean())
            result[f"{selector}_{metric}_std"] = float(values.std())
    result["harm_beats_beta_actual_count"] = sum(
        next(row for row in record["direction_rows"] if row["selector"] == "harm")["actual_centered_harm"]
        > next(row for row in record["direction_rows"] if row["selector"] == "beta")["actual_centered_harm"]
        for record in records
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--feature-epochs", type=int, default=DEFAULT_FEATURE_EPOCHS)
    parser.add_argument("--head-steps", type=int, default=DEFAULT_HEAD_STEPS)
    parser.add_argument("--irm-lambda", type=float, default=DEFAULT_IRM_LAMBDA)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--random-directions", type=int, default=DEFAULT_RANDOM_DIRECTIONS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = [
        run_seed(
            seed,
            args.n_per_env,
            args.feature_epochs,
            args.head_steps,
            args.irm_lambda,
            args.epsilon,
            args.lambda_reg,
            args.damping,
            args.random_directions,
        )
        for seed in args.seeds
    ]
    output = {
        "configuration": vars(args) | {"json": str(args.json) if args.json else None},
        "scope": "frozen_classifier_head_source_contrasts",
        "target_used": False,
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
