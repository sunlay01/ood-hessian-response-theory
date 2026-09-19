"""Population response-matrix witness for curvature-corrected aggregation.

Two source risks are locally represented at theta=0 by gradients g_e and
curvatures H_e.  Candidate steps are d_e=-alpha*g_e.  The witness is chosen
so that the first-order response of every candidate is negative, while the
large step d_1 has a positive second-order response on source 2:

    g_2 d_1 < 0 but g_2 d_1 + .5 H_2 d_1^2 > 0.

The first-order aggregator cannot see this.  CCRA optimizes the convex weight
over candidate steps using the actual quadratic response of the combined
step.  A target with the same response direction but slightly larger curvature
tests whether the correction transfers beyond the observed source pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


torch.set_num_threads(1)


def risk(theta: torch.Tensor, gradient: float, hessian: float) -> torch.Tensor:
    return gradient * theta + 0.5 * hessian * theta.pow(2)


def optimize_weights(
    gradients: torch.Tensor,
    hessians: torch.Tensor,
    candidates: torch.Tensor,
    use_curvature: bool,
    response_lambda: float,
    steps: int,
    lr: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.zeros(candidates.shape[0], requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    uniform = torch.full_like(logits, 1.0 / logits.numel())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        update = torch.dot(weights, candidates)
        first = gradients * update
        response = first + 0.5 * hessians * update.pow(2) if use_curvature else first
        penalty = torch.nn.functional.softplus(response / temperature).pow(2).mean()
        objective = 0.5 * (weights - uniform).pow(2).sum() + response_lambda * penalty
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        weights = torch.softmax(logits, dim=0)
        update = torch.dot(weights, candidates)
        first = gradients * update
        response = first + 0.5 * hessians * update.pow(2)
    return weights, update, torch.stack([first, response])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--target-hessian", type=float, default=3.5)
    parser.add_argument("--response-lambda", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    gradients = torch.tensor([-1.0, -0.2])
    hessians = torch.tensor([1.0, 3.0])
    candidates = -args.alpha * gradients
    first_matrix = gradients[:, None] * candidates[None, :]
    second_matrix = first_matrix + 0.5 * hessians[:, None] * candidates[None, :].pow(2)

    first_weights, first_update, first_response = optimize_weights(
        gradients,
        hessians,
        candidates,
        False,
        args.response_lambda,
        args.steps,
        args.lr,
        args.temperature,
    )
    curvature_weights, curvature_update, curvature_response = optimize_weights(
        gradients,
        hessians,
        candidates,
        True,
        args.response_lambda,
        args.steps,
        args.lr,
        args.temperature,
    )

    target_gradient = -0.2
    target_hessian = args.target_hessian
    target_star = -target_gradient / target_hessian
    target_at_zero = float(risk(torch.tensor(0.0), target_gradient, target_hessian))
    rows = []
    for method, update in (
        ("first_order", first_update),
        ("curvature_corrected", curvature_update),
    ):
        target_risk = float(risk(update, target_gradient, target_hessian))
        optimum_risk = float(risk(torch.tensor(target_star), target_gradient, target_hessian))
        rows.append({
            "method": method,
            "update": float(update),
            "target_risk": target_risk,
            "target_excess": target_risk - optimum_risk,
            "target_response_from_zero": target_risk - target_at_zero,
        })
    output = {
        "configuration": vars(args) | {"json": str(args.json) if args.json else None},
        "source_gradients": gradients.tolist(),
        "source_hessians": hessians.tolist(),
        "candidates": candidates.tolist(),
        "first_order_response_matrix": first_matrix.tolist(),
        "second_order_response_matrix": second_matrix.tolist(),
        "sign_reversal_entries": int(((first_matrix < 0) & (second_matrix > 0)).sum()),
        "first_order_weights": first_weights.tolist(),
        "curvature_weights": curvature_weights.tolist(),
        "first_order_combined_response": first_response.tolist(),
        "curvature_combined_response": curvature_response.tolist(),
        "rows": rows,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
