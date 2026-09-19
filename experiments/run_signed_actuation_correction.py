"""Residual-regime sweep for signed IRMv1 actuation correction.

At each solution of R_source + alpha * Omega_IRM, this script computes the
correct solution-path actuation

    v_alpha = (H_R + alpha H_Omega + damping I)^-1 grad Omega,

the worst signed blind direction, and the minimum-norm single-constraint
correction

    u = q_+(h_harm) / ||G_g h_harm||^2 * G_g h_harm.

The positive sign is required by the constraint
    -<G_g h_harm, v_alpha + u> <= 0.

Selection is source-only.  A held-out target is evaluated only after the local
step, and is never used to choose the correction or any hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_actuation_harm_validation import (
    frozen_environment_data,
    head_objective,
    head_risk,
    mixture_risk,
    train_feature_model,
)
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


def optimize_head_lbfgs(
    start: torch.Tensor,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    max_iter: int,
) -> torch.Tensor:
    head = start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [head],
        lr=0.5,
        max_iter=max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = head_objective(head, envs, alpha)
        objective.backward()
        return objective

    optimizer.step(closure)
    return head.detach()


def path_geometry(
    head: torch.Tensor,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    alpha: float,
    damping: float,
) -> dict[str, object]:
    variable = head.detach().clone().requires_grad_(True)

    def source_risk(value: torch.Tensor) -> torch.Tensor:
        return torch.stack([head_risk(value, env) for env in envs]).mean()

    def omega(value: torch.Tensor) -> torch.Tensor:
        residuals = []
        for feature, label in envs:
            logits = feature @ value[:-1] + value[-1]
            residuals.append(irm_statistics(logits, label, create_graph=True))
        return torch.stack(residuals).pow(2).mean()

    residuals = []
    gradients = []
    for feature, label in envs:
        logits = feature @ variable[:-1] + variable[-1]
        residuals.append(irm_statistics(logits, label, create_graph=True))
        gradients.append(torch.autograd.grad(head_risk(variable, (feature, label)), variable, retain_graph=True)[0])
    residual_tensor = torch.stack(residuals)
    omega_value = residual_tensor.pow(2).mean()
    grad_omega = torch.autograd.grad(omega_value, variable, create_graph=False)[0]
    hessian_risk = torch.autograd.functional.hessian(source_risk, variable)
    hessian_omega = torch.autograd.functional.hessian(omega, variable)
    total_hessian = hessian_risk + alpha * hessian_omega
    identity = torch.eye(total_hessian.shape[0], dtype=total_hessian.dtype)
    actuation = torch.linalg.solve(total_hessian + damping * identity, grad_omega)

    gradient_matrix = torch.stack(gradients, dim=1).detach().cpu().numpy()
    g_map = gradient_matrix @ contrasts
    observation = residual_tensor.detach().cpu().numpy() @ contrasts
    observation = observation.reshape(1, -1)
    projector = np.eye(observation.shape[1]) - np.linalg.pinv(
        observation, rcond=1e-8
    ) @ observation
    projector = (projector + projector.T) / 2.0
    v = actuation.detach().cpu().numpy()
    harmful_covector = projector @ g_map.T @ v
    chi = float(np.linalg.norm(harmful_covector))
    h_harm = -harmful_covector / max(chi, 1e-12)
    risk_gradient = g_map @ h_harm
    q = -float(risk_gradient @ v)
    correction = (
        max(q, 0.0)
        / max(float(risk_gradient @ risk_gradient), 1e-12)
        * risk_gradient
    )
    corrected = v + correction
    q_corrected = -float(risk_gradient @ corrected)
    return {
        "residual_rms": float(torch.sqrt(residual_tensor.pow(2).mean()).detach()),
        "objective_gradient_norm": float(
            torch.linalg.norm(
                torch.stack(gradients, dim=1).mean(dim=1) + alpha * grad_omega
            ).detach()
        ),
        "actuation": v,
        "actuation_norm": float(np.linalg.norm(v)),
        "chi": chi,
        "h_harm": h_harm,
        "risk_gradient": risk_gradient,
        "q": q,
        "correction": correction,
        "correction_norm": float(np.linalg.norm(correction)),
        "corrected_actuation": corrected,
        "q_corrected": q_corrected,
    }


def target_head_risk(
    model: MLP,
    target: tuple[torch.Tensor, torch.Tensor],
    head: np.ndarray,
) -> float:
    feature = model.feature(target[0]).detach()
    vector = torch.tensor(head, dtype=feature.dtype)
    return float(head_risk(vector, (feature, target[1])).detach())


def local_step_metrics(
    model: MLP,
    target: tuple[torch.Tensor, torch.Tensor],
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    head: np.ndarray,
    geometry: dict[str, object],
    epsilon: float,
    delta_alpha: float,
) -> dict[str, float]:
    base_weights = np.ones(SOURCE_ENVS) / SOURCE_ENVS
    h_harm = np.asarray(geometry["h_harm"])
    shifted_weights = base_weights + epsilon * (contrasts @ h_harm)
    if float(np.min(shifted_weights)) < -1e-10:
        raise ValueError("epsilon leaves the source-mixture simplex")
    original = head - delta_alpha * np.asarray(geometry["actuation"])
    corrected = head - delta_alpha * np.asarray(geometry["corrected_actuation"])

    def centered_harm(stepped: np.ndarray) -> float:
        shift_change = mixture_risk(stepped, envs, shifted_weights) - mixture_risk(
            head, envs, shifted_weights
        )
        base_change = mixture_risk(stepped, envs, base_weights) - mixture_risk(
            head, envs, base_weights
        )
        return shift_change - base_change

    source_original = mixture_risk(original, envs, base_weights)
    source_corrected = mixture_risk(corrected, envs, base_weights)
    return {
        "original_centered_harm": centered_harm(original),
        "corrected_centered_harm": centered_harm(corrected),
        "harm_reduction": centered_harm(original) - centered_harm(corrected),
        "source_risk_original": source_original,
        "source_risk_corrected": source_corrected,
        "source_risk_correction_cost": source_corrected - source_original,
        "target_risk_original": target_head_risk(model, target, original),
        "target_risk_corrected": target_head_risk(model, target, corrected),
        "target_risk_correction_gain": target_head_risk(model, target, original)
        - target_head_risk(model, target, corrected),
    }


def run_seed(
    seed: int,
    n_per_env: int,
    feature_epochs: int,
    head_iterations: int,
    strengths: list[float],
    epsilon: float,
    delta_alpha: float,
    damping: float,
) -> dict[str, object]:
    set_seed(seed)
    data_geometry = make_geometry(seed)
    source, _, target, _, _ = make_datasets(
        data_geometry, n_per_env, gamma=1.0, seed=seed
    )
    model = train_feature_model(source, feature_epochs, DEFAULT_LR)
    envs = frozen_environment_data(model, source)
    contrasts = contrast_matrix(SOURCE_ENVS)
    head = torch.cat(
        [model.fc2.weight.detach().reshape(-1), model.fc2.bias.detach().reshape(-1)]
    )
    rows = []
    for alpha in strengths:
        head = optimize_head_lbfgs(head, envs, alpha, head_iterations)
        geometry = path_geometry(head, envs, contrasts, alpha, damping)
        metrics = local_step_metrics(
            model,
            target,
            envs,
            contrasts,
            head.detach().cpu().numpy(),
            geometry,
            epsilon,
            delta_alpha,
        )
        rows.append(
            {
                "strength": alpha,
                "residual_rms": geometry["residual_rms"],
                "objective_gradient_norm": geometry["objective_gradient_norm"],
                "actuation_norm": geometry["actuation_norm"],
                "chi": geometry["chi"],
                "q": geometry["q"],
                "q_corrected": geometry["q_corrected"],
                "correction_norm": geometry["correction_norm"],
            }
            | metrics
        )
    return {"seed": seed, "rows": rows}


def summarize(records: list[dict[str, object]], strengths: list[float]) -> list[dict[str, float]]:
    summary = []
    for alpha in strengths:
        rows = [
            next(row for row in record["rows"] if row["strength"] == alpha)
            for record in records
        ]
        item = {"strength": alpha}
        for metric in (
            "residual_rms",
            "objective_gradient_norm",
            "actuation_norm",
            "chi",
            "q_corrected",
            "correction_norm",
            "harm_reduction",
            "source_risk_correction_cost",
            "target_risk_correction_gain",
        ):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_std"] = float(values.std())
        item["target_gain_positive_count"] = sum(
            row["target_risk_correction_gain"] > 0 for row in rows
        )
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--feature-epochs", type=int, default=40)
    parser.add_argument("--head-iterations", type=int, default=300)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--delta-alpha", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = [
        run_seed(
            seed,
            args.n_per_env,
            args.feature_epochs,
            args.head_iterations,
            args.strengths,
            args.epsilon,
            args.delta_alpha,
            args.damping,
        )
        for seed in args.seeds
    ]
    output = {
        "configuration": vars(args) | {"json": str(args.json) if args.json else None},
        "selection_uses_target": False,
        "correction_sign": "positive",
        "records": records,
        "summary": summarize(records, args.strengths),
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
