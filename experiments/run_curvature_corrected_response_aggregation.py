"""Curvature-Corrected Response Aggregation (CCRA).

This probe implements the narrow idea in the response-matrix proposal.  For
each source environment e, the candidate update is d_e = -alpha g_e.  Its
cross-environment response matrix is

    M^(1)_{je} = g_j^T d_e,
    M^(2)_{je} = g_j^T d_e + 1/2 d_e^T H_j d_e.

CCRA chooses a convex combination d(w) of the candidate updates by penalizing
positive predicted responses on source environments.  The first-order control
uses M^(1); CCRA uses the finite-step curvature-corrected response M^(2) of the
combined update itself.  This is a response-sign experiment, not a beta,
IRM, or Hessian-alignment objective.

The encoder is warmed up once and frozen.  The classifier head uses exact
per-environment logistic gradients and Hessians.  Target data are used only
for final evaluation.
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
    gradients = []
    hessians = []
    for feature, label in zip(features, labels):
        augmented = torch.cat(
            [feature, torch.ones((feature.shape[0], 1), dtype=feature.dtype)], dim=1
        )
        logits = augmented @ head
        probability = torch.sigmoid(logits)
        gradients.append(augmented.T @ (probability - label) / feature.shape[0])
        weights = probability * (1.0 - probability)
        hessians.append(augmented.T @ (weights[:, None] * augmented) / feature.shape[0])
    return torch.stack(gradients), torch.stack(hessians)


def response_matrix(head, features, labels, alpha):
    gradients, hessians = per_environment_geometry(head, features, labels)
    candidates = -alpha * gradients
    first = gradients @ candidates.T
    second = first + 0.5 * torch.einsum(
        "ei,jik,ek->je", candidates, hessians, candidates
    )
    return gradients, hessians, candidates, first, second


def combined_response(gradients, hessians, candidates, weights, use_curvature):
    update = weights @ candidates
    first = gradients @ update
    if use_curvature:
        second = first + 0.5 * torch.einsum(
            "eij,i,j->e", hessians, update, update
        )
    else:
        second = first
    return update, first, second


def simplex_response_update(
    gradients,
    hessians,
    candidates,
    alpha,
    use_curvature,
    inner_steps,
    inner_lr,
    response_lambda,
    temperature,
):
    # The inner simplex solve selects an update using the frozen local
    # response geometry.  It must not retain the outer head autograd graph.
    gradients = gradients.detach()
    hessians = hessians.detach()
    candidates = candidates.detach()
    count = candidates.shape[0]
    logits = torch.zeros(count, dtype=candidates.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=inner_lr)
    uniform = torch.full((count,), 1.0 / count, dtype=candidates.dtype)
    for _ in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        update, first, predicted = combined_response(
            gradients, hessians, candidates, weights, use_curvature
        )
        positive = F.softplus(predicted / temperature).pow(2).mean()
        objective = 0.5 * (weights - uniform).pow(2).sum() + response_lambda * positive
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        weights = torch.softmax(logits, dim=0)
        update, first, predicted = combined_response(
            gradients, hessians, candidates, weights, use_curvature
        )
    return update.detach(), weights.detach(), first.detach(), predicted.detach()


def vrex_step(head, features, labels, lr, vrex_lambda):
    value = head.detach().clone().requires_grad_(True)
    risks = head_risks(value, features, labels)
    objective = risks.mean() + vrex_lambda * (risks - risks.mean()).pow(2).mean()
    gradient = torch.autograd.grad(objective, value)[0]
    return (value - lr * gradient).detach()


def run_method(
    method,
    initial,
    features,
    labels,
    epochs,
    alpha,
    inner_steps,
    inner_lr,
    response_lambda,
    temperature,
    vrex_lambda,
):
    head = initial.clone()
    histories = []
    for _ in range(epochs):
        if method == "erm":
            gradients, hessians, candidates, first_matrix, second_matrix = response_matrix(
                head, features, labels, alpha
            )
            weights = torch.full(
                (candidates.shape[0],), 1.0 / candidates.shape[0], dtype=head.dtype
            )
            update, first, predicted = combined_response(
                gradients, hessians, candidates, weights, True
            )
        elif method == "vrex":
            head = vrex_step(head, features, labels, alpha, vrex_lambda)
            gradients, hessians, candidates, first_matrix, second_matrix = response_matrix(
                head, features, labels, alpha
            )
            weights = torch.full(
                (candidates.shape[0],), 1.0 / candidates.shape[0], dtype=head.dtype
            )
            update, first, predicted = combined_response(
                gradients, hessians, candidates, weights, True
            )
            histories.append({
                "sign_reversals": int(((first_matrix < 0) & (second_matrix > 0)).sum()),
                "max_first": float(first_matrix.max().detach()),
                "max_second": float(second_matrix.max().detach()),
            })
            continue
        else:
            gradients, hessians, candidates, first_matrix, second_matrix = response_matrix(
                head, features, labels, alpha
            )
            update, weights, first, predicted = simplex_response_update(
                gradients,
                hessians,
                candidates,
                alpha,
                method == "curvature_response",
                inner_steps,
                inner_lr,
                response_lambda,
                temperature,
            )
        head = (head + update).detach()
        histories.append({
            "sign_reversals": int(((first_matrix < 0) & (second_matrix > 0)).sum()),
            "max_first": float(first_matrix.max().detach()),
            "max_second": float(second_matrix.max().detach()),
            "combined_first_max": float(first.max().detach()),
            "combined_second_max": float(predicted.max().detach()),
            "weight_entropy": float(-(weights * weights.clamp_min(1e-12).log()).sum()),
        })
    return head, histories


def run_one(
    seed,
    n_per_env,
    gamma,
    warmup_epochs,
    epochs,
    feature_lr,
    alpha,
    inner_steps,
    inner_lr,
    response_lambda,
    temperature,
    vrex_lambda,
    target_shift_mode,
):
    set_seed(seed)
    geometry = make_geometry(seed)
    source, _, target, _, _ = make_datasets(
        geometry, n_per_env, gamma, seed, target_shift_mode=target_shift_mode
    )
    warmup = MLP()
    optimizer = torch.optim.Adam(warmup.parameters(), lr=feature_lr)
    for _ in range(warmup_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [F.binary_cross_entropy_with_logits(warmup(x), y) for x, y in source]
        ).mean()
        loss.backward()
        optimizer.step()
    features = [warmup.feature(x).detach() for x, _ in source]
    labels = [y for _, y in source]
    initial = head_vector(warmup)
    rows = []
    for method in ("erm", "vrex", "first_order_response", "curvature_response"):
        head, history = run_method(
            method,
            initial,
            features,
            labels,
            epochs,
            alpha,
            inner_steps,
            inner_lr,
            response_lambda,
            temperature,
            vrex_lambda,
        )
        model = MLP()
        model.load_state_dict(copy.deepcopy(warmup.state_dict()))
        with torch.no_grad():
            model.fc2.weight.copy_(head[:-1].reshape_as(model.fc2.weight))
            model.fc2.bias.copy_(head[-1:].reshape_as(model.fc2.bias))
        source_eval = evaluate(
            model,
            (torch.cat([x for x, _ in source]), torch.cat([y for _, y in source])),
        )
        target_eval = evaluate(model, target)
        rows.append({
            "method": method,
            "source_loss": source_eval["loss"],
            "source_accuracy": source_eval["accuracy"],
            "target_loss": target_eval["loss"],
            "target_accuracy": target_eval["accuracy"],
            "mean_sign_reversals": float(np.mean([x["sign_reversals"] for x in history])),
            "max_sign_reversals": max(x["sign_reversals"] for x in history),
            "mean_combined_second_max": float(np.mean([x.get("combined_second_max", 0.0) for x in history])),
            "mean_weight_entropy": float(np.mean([x.get("weight_entropy", 0.0) for x in history])),
        })
    return {
        "seed": seed,
        "n_per_env": n_per_env,
        "gamma": gamma,
        "target_shift_mode": target_shift_mode,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--feature-lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--inner-lr", type=float, default=0.08)
    parser.add_argument("--response-lambda", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.01)
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
            args.alpha,
            args.inner_steps,
            args.inner_lr,
            args.response_lambda,
            args.temperature,
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
