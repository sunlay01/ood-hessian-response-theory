"""Hessian-Response Regularization (HRR) controlled experiment.

HRR is the direct algorithmic instantiation of the local target-risk bridge:

  R_T - R_S <= rho ||Delta g|| + rho^2/2 ||Delta H|| + O(rho^3).

At each source-only training step, it forms the centered environment maps

  G_g  = [g_e-g_bar] C,
  G_H  = [H_e-H_bar] C,

where C is an orthonormal source-environment contrast basis.  The objective is

  mean_e R_e + lambda_g rho ||G_g||_F
                 + lambda_H rho^2/2 ||G_H||_F.

The Hessian term is the proposed mechanism.  ``gradient_response`` removes it
and is the signal-removal ablation.  ERM and V-REx are controls.  The encoder
is warmed up once and frozen; the classifier head is trained with exact
per-environment logistic gradients and Hessians, so the experiment is a clean
test of the theory rather than a Hessian-sketch implementation artifact.

No target samples, target labels, latent target coordinates, or IRM statistic
are used by HRR.  IRMv1 is intentionally not part of the proposed method.
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
    DEFAULT_LR,
    MLP,
    SOURCE_ENVS,
    contrast_matrix,
    evaluate,
    make_datasets,
    make_geometry,
    set_seed,
)


torch.set_num_threads(1)


def head_vector(model: MLP) -> torch.Tensor:
    return torch.cat([model.fc2.weight.reshape(-1), model.fc2.bias.reshape(-1)])


def head_risks(head: torch.Tensor, features, labels) -> torch.Tensor:
    return torch.stack(
        [
            F.binary_cross_entropy_with_logits(
                feature @ head[:-1] + head[-1], label
            )
            for feature, label in zip(features, labels)
        ]
    )


def per_environment_geometry(head: torch.Tensor, features, labels):
    """Analytic logistic head gradient/Hessian, differentiable in ``head``."""
    gradients = []
    hessians = []
    for feature, label in zip(features, labels):
        augmented = torch.cat(
            [feature, torch.ones((feature.shape[0], 1), dtype=feature.dtype)], dim=1
        )
        logits = augmented @ head
        probability = torch.sigmoid(logits)
        gradient = (augmented.T @ (probability - label)) / feature.shape[0]
        weights = probability * (1.0 - probability)
        hessian = augmented.T @ (weights[:, None] * augmented) / feature.shape[0]
        gradients.append(gradient)
        hessians.append(hessian)
    return torch.stack(gradients), torch.stack(hessians)


def response_penalties(
    head: torch.Tensor,
    features,
    labels,
    contrasts: torch.Tensor,
    rho: float,
    include_gradient: bool,
    include_hessian: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    gradients, hessians = per_environment_geometry(head, features, labels)
    mean_gradient = gradients.mean(dim=0)
    mean_hessian = hessians.mean(dim=0)
    centered_gradient = gradients - mean_gradient
    centered_hessian = hessians - mean_hessian
    gradient_map = centered_gradient.T @ contrasts
    hessian_map = torch.einsum("eij,ek->kij", centered_hessian, contrasts)
    gradient_norm = torch.linalg.matrix_norm(gradient_map, ord="fro")
    # The theorem uses the operator norm in parameter space.  We aggregate
    # the contrast-coordinate operator norms in l2; replacing this by the
    # Frobenius norm is a valid but looser signal-removal ablation.
    hessian_op_by_contrast = torch.linalg.matrix_norm(hessian_map, ord=2, dim=(-2, -1))
    hessian_norm = torch.linalg.vector_norm(hessian_op_by_contrast)
    gradient_component = rho * gradient_norm if include_gradient else torch.zeros_like(gradient_norm)
    hessian_component = 0.5 * rho * rho * hessian_norm if include_hessian else torch.zeros_like(hessian_norm)
    value = gradient_component + hessian_component
    return value, {
        "gradient_response": float(gradient_norm.detach()),
        "hessian_response": float(hessian_norm.detach()),
        "weighted_gradient_response": float((rho * gradient_norm).detach()),
        "weighted_hessian_response": float((0.5 * rho * rho * hessian_norm).detach()),
    }, gradient_component, hessian_component


def objective(
    model: MLP,
    features,
    labels,
    contrasts: torch.Tensor,
    rho: float,
    lambda_g: float,
    lambda_h: float,
    vrex_lambda: float,
    include_gradient: bool,
    include_hessian: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    head = head_vector(model)
    risks = head_risks(head, features, labels)
    source = risks.mean()
    response, response_info, gradient_component, hessian_component = response_penalties(
        head,
        features,
        labels,
        contrasts,
        rho,
        include_gradient,
        include_hessian,
    )
    # V-REx control is kept separate from the response objective.
    vrex = (risks - source).pow(2).mean()
    total = source + lambda_g * gradient_component + lambda_h * hessian_component + vrex_lambda * vrex
    info = {
        "source_loss": float(source.detach()),
        "vrex_penalty": float(vrex.detach()),
        "objective": float(total.detach()),
        **response_info,
    }
    return total, info


def train_method(
    warmup: MLP,
    features,
    labels,
    contrasts: torch.Tensor,
    method: str,
    epochs: int,
    lr: float,
    rho: float,
    lambda_g: float,
    lambda_h: float,
    vrex_lambda: float,
) -> tuple[MLP, dict[str, float]]:
    model = MLP()
    model.load_state_dict(copy.deepcopy(warmup.state_dict()))
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=lr)
    last: dict[str, float] = {}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if method == "erm":
            total, info = objective(
                model, features, labels, contrasts, rho, 0.0, 0.0, 0.0, False, False
            )
        elif method == "vrex":
            total, info = objective(
                model, features, labels, contrasts, rho, 0.0, 0.0, vrex_lambda, False, False
            )
        elif method == "gradient_response":
            total, info = objective(
                model, features, labels, contrasts, rho, lambda_g, 0.0, 0.0, True, False
            )
        elif method == "hrr":
            total, info = objective(
                model, features, labels, contrasts, rho, lambda_g, lambda_h, 0.0, True, True
            )
        elif method == "hessian_only":
            total, info = objective(
                model, features, labels, contrasts, rho, 0.0, lambda_h, 0.0, False, True
            )
        else:
            raise ValueError(method)
        total.backward()
        optimizer.step()
        last = info
    return model, last


def run_one(
    seed: int,
    n_per_env: int,
    gamma: float,
    warmup_epochs: int,
    epochs: int,
    feature_lr: float,
    head_lr: float,
    rho: float,
    lambda_g: float,
    lambda_h: float,
    vrex_lambda: float,
    target_shift_mode: str,
) -> dict[str, object]:
    set_seed(seed)
    geometry = make_geometry(seed)
    source, _, target, _, _ = make_datasets(
        geometry, n_per_env, gamma, seed, target_shift_mode=target_shift_mode
    )
    warmup = MLP()
    feature_optimizer = torch.optim.Adam(warmup.parameters(), lr=feature_lr)
    for _ in range(warmup_epochs):
        feature_optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [F.binary_cross_entropy_with_logits(warmup(x), y) for x, y in source]
        ).mean()
        loss.backward()
        feature_optimizer.step()
    features = [warmup.feature(x).detach() for x, _ in source]
    labels = [y for _, y in source]
    contrasts = torch.tensor(contrast_matrix(SOURCE_ENVS), dtype=torch.float32)
    rows = []
    for method in ("erm", "vrex", "gradient_response", "hessian_only", "hrr"):
        model, train_info = train_method(
            warmup,
            features,
            labels,
            contrasts,
            method,
            epochs,
            head_lr,
            rho,
            lambda_g,
            lambda_h,
            vrex_lambda,
        )
        source_metric = evaluate(
            model,
            (
                torch.cat([x for x, _ in source]),
                torch.cat([y for _, y in source]),
            ),
        )
        target_metric = evaluate(model, target)
        rows.append(
            {
                "method": method,
                "source_loss": source_metric["loss"],
                "source_accuracy": source_metric["accuracy"],
                "target_loss": target_metric["loss"],
                "target_accuracy": target_metric["accuracy"],
                **train_info,
            }
        )
    return {
        "seed": seed,
        "n_per_env": n_per_env,
        "gamma": gamma,
        "target_shift_mode": target_shift_mode,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--feature-lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--head-lr", type=float, default=0.03)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--lambda-g", type=float, default=1.0)
    parser.add_argument("--lambda-h", type=float, default=1.0)
    parser.add_argument("--vrex-lambda", type=float, default=1.0)
    parser.add_argument(
        "--target-shift-mode",
        choices=["predictive", "nuisance", "sign_reversal"],
        default="predictive",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = [
        run_one(
            seed,
            args.n_per_env,
            gamma,
            args.warmup_epochs,
            args.epochs,
            args.feature_lr,
            args.head_lr,
            args.rho,
            args.lambda_g,
            args.lambda_h,
            args.vrex_lambda,
            args.target_shift_mode,
        )
        for gamma in args.gammas
        for seed in args.seeds
    ]
    configuration = vars(args).copy()
    configuration["json"] = str(configuration["json"]) if configuration["json"] else None
    output = {"configuration": configuration, "records": records}
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
