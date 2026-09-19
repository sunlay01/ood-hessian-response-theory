"""Population witness where Hessian response is action-relevant.

The one-dimensional risk family is

  R_a(theta) = .5 (theta-1)^2 + a log(cosh(theta)),

with source environment coefficients a in {-A,+A}.  The source mean risk is
independent of a, so ERM selects theta=1.  However

  g_a(theta) = theta-1 + a tanh(theta),
  H_a(theta) = 1 + a sech(theta)^2.

At theta near zero the gradient response can be small while the Hessian
response is nonzero.  HRR minimizes the exact local bridge proxy

  R_bar(theta) + lambda_g rho ||G_g(theta)||
                    + lambda_H rho^2/2 ||G_H(theta)||.

The target coefficient is chosen from the source-covered direction (a=-A or
a=+A), so this is a source-only, identifiable local transfer test.  The
experiment reports the local certificate and target risk, and includes a
gradient-only signal-removal ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


torch.set_num_threads(1)


def risk(theta: torch.Tensor, a: float) -> torch.Tensor:
    return 0.5 * (theta - 1.0).pow(2) + a * torch.log(torch.cosh(theta))


def target_optimum(a: float) -> float:
    grid = torch.linspace(-4.0, 6.0, 200001)
    values = risk(grid, a)
    return float(grid[torch.argmin(values)])


def certificate(theta: torch.Tensor, source_a: torch.Tensor, rho: float):
    gradient_response = source_a.std(unbiased=False) * torch.tanh(theta).abs()
    hessian_response = source_a.std(unbiased=False) * (1.0 / torch.cosh(theta).pow(2)).abs()
    return gradient_response, hessian_response


def optimize(
    source_a: torch.Tensor,
    target_a: float,
    method: str,
    rho: float,
    lambda_g: float,
    lambda_h: float,
    steps: int,
    lr: float,
    start: float,
) -> dict[str, float]:
    theta = torch.tensor(float(start), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        source = torch.stack([risk(theta, float(a)) for a in source_a]).mean()
        gradient_response, hessian_response = certificate(theta, source_a, rho)
        use_g = method in {"gradient_response", "hrr"}
        use_h = method in {"hessian_response", "hrr"}
        objective = source
        if use_g:
            objective = objective + lambda_g * rho * gradient_response
        if use_h:
            objective = objective + lambda_h * 0.5 * rho * rho * hessian_response
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        source_risk = torch.stack([risk(theta, float(a)) for a in source_a]).mean()
        target_risk = risk(theta, target_a)
        grad_resp, hess_resp = certificate(theta, source_a, rho)
        target_star = risk(torch.tensor(target_optimum(target_a)), target_a)
    return {
        "method": method,
        "theta": float(theta.detach()),
        "source_risk": float(source_risk),
        "target_risk": float(target_risk),
        "target_excess": float(target_risk - target_star),
        "gradient_response": float(grad_resp),
        "hessian_response": float(hess_resp),
        "target_optimum": target_optimum(target_a),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.4)
    parser.add_argument("--target-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--source-design", choices=["symmetric", "monotone"], default="symmetric")
    parser.add_argument("--target-multiplier", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--lambda-g", type=float, default=0.2)
    parser.add_argument("--lambda-h", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.source_design == "symmetric":
        source_a = torch.tensor([-args.a, args.a], dtype=torch.float32)
        target_a = float(args.target_sign * args.target_multiplier * args.a)
    else:
        source_a = torch.tensor([0.0, args.a], dtype=torch.float32)
        target_a = float(args.target_multiplier * args.a)
    rows = [
        optimize(
            source_a,
            target_a,
            method,
            args.rho,
            args.lambda_g,
            args.lambda_h,
            args.steps,
            args.lr,
            args.start,
        )
        for method in ("erm", "gradient_response", "hessian_response", "hrr")
    ]
    output = {"configuration": vars(args) | {"json": str(args.json) if args.json else None}, "rows": rows}
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
