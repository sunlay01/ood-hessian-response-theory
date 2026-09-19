"""End-to-end CCRA with a classifier-head virtual step.

The inner selector uses only classifier-head gradients/Hessians on an inner
batch.  Its selected step is detached before the outer loss is evaluated on a
disjoint batch:

    w_tilde = w + stop_gradient(d_star)
    L = mean_e R_e^out(psi, w)
        + gamma * mean_e [R_e^out(psi,w_tilde)-R_e^out(psi,w)]_+^2

E2E-FO uses the same selector with the Hessian term removed.  The distinction
is therefore exactly the curvature correction in the response model.  ERM and
V-REx are source-only controls.  The encoder is updated by the outer loss;
the head Hessian is never formed for encoder parameters.
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


class CurvatureModel(torch.nn.Module):
    """Fixed two-feature model for a source-covered curvature witness."""

    def __init__(self) -> None:
        super().__init__()
        self.fc2 = torch.nn.Linear(2, 1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.feature(x)).squeeze(-1)


def make_curvature_datasets(
    n_per_env: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """Create source data with a genuine first/second-order sign reversal.

    At the zero head, envs 1 and 2 have gradients in e1/e2.  Env 3 has a
    small positive mean gradient but a large e1 covariance.  With alpha=1.5,
    the e1 candidate has negative linear response yet positive quadratic and
    exact outer response on env 3.  This is a source-observable curvature
    shift, not an evaluation-only target construction.
    """
    count = max(int(n_per_env), 4)
    half = count // 2
    env1 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32).repeat(half, 1)
    y1 = torch.tensor([0.0, 1.0], dtype=torch.float32).repeat(half)
    env2 = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=torch.float32).repeat(half, 1)
    y2 = torch.tensor([0.0, 1.0], dtype=torch.float32).repeat(half)
    pattern = torch.tensor([[4.2, 0.2], [-3.8, 0.2]], dtype=torch.float32)
    env3 = pattern.repeat((count + 1) // 2, 1)[:count]
    # Small opposite residuals keep the bias gradient zero while preserving a
    # large x1 covariance.  Labels are soft targets only to make the witness
    # exactly source-covered at the zero head.
    residual = 0.025
    y3 = torch.tensor([0.5 - residual, 0.5 + residual], dtype=torch.float32).repeat(
        (count + 1) // 2
    )[:count]
    # Duplicate the mechanism with independent tensors.  The first triplet is
    # available to the inner selector; the second triplet is held out for the
    # environment-level outer objective.
    source = [
        (env1, y1),
        (env2, y2),
        (env3, y3),
        (env1.clone(), y1.clone()),
        (env2.clone(), y2.clone()),
        (env3.clone(), y3.clone()),
    ]
    return source, (env3.clone(), y3.clone())


def head_vector(model: MLP) -> torch.Tensor:
    return torch.cat([model.fc2.weight.reshape(-1), model.fc2.bias.reshape(-1)])


def set_head(model: MLP, head: torch.Tensor) -> None:
    with torch.no_grad():
        model.fc2.weight.copy_(head[:-1].reshape_as(model.fc2.weight))
        model.fc2.bias.copy_(head[-1:].reshape_as(model.fc2.bias))


def custom_risk(model: MLP, x: torch.Tensor, y: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    feature = model.feature(x)
    return F.binary_cross_entropy_with_logits(feature @ head[:-1] + head[-1], y)


def head_geometry(head: torch.Tensor, features: list[torch.Tensor], labels: list[torch.Tensor]):
    gradients, hessians = [], []
    for feature, label in zip(features, labels):
        augmented = torch.cat(
            [feature, torch.ones((feature.shape[0], 1), dtype=feature.dtype)], dim=1
        )
        logits = augmented @ head
        probabilities = torch.sigmoid(logits)
        gradients.append(augmented.T @ (probabilities - label) / feature.shape[0])
        weights = probabilities * (1.0 - probabilities)
        hessians.append(augmented.T @ (weights[:, None] * augmented) / feature.shape[0])
    return torch.stack(gradients), torch.stack(hessians)


def select_virtual_step(
    head: torch.Tensor,
    inner_features: list[torch.Tensor],
    inner_labels: list[torch.Tensor],
    alpha: float,
    use_curvature: bool,
    selector_steps: int,
    selector_lr: float,
    selector_lambda: float,
    selector_temperature: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    gradients, hessians = head_geometry(head.detach(), inner_features, inner_labels)
    gradient_norms = gradients.norm(dim=1, keepdim=True).clamp_min(1e-8)
    candidates = -alpha * gradients / gradient_norms
    candidate_first = gradients @ candidates.T
    candidate_second = candidate_first + 0.5 * torch.einsum(
        "eij,kj,ki->ek", hessians, candidates, candidates
    )
    temperature = max(float(selector_temperature), 1e-6)
    logits = torch.zeros(len(inner_features), dtype=head.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=selector_lr)
    uniform = torch.full_like(logits, 1.0 / logits.numel())
    for _ in range(selector_steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits / temperature, dim=0)
        update = weights @ candidates
        first = gradients @ update
        predicted = first
        if use_curvature:
            predicted = predicted + 0.5 * torch.einsum(
                "eij,i,j->e", hessians, update, update
            )
        # A negative response is already safe. Penalize only the harmful
        # positive part; a softplus gate would keep pushing safe updates.
        violation = F.relu(predicted).pow(2).mean()
        objective = 0.5 * (weights - uniform).pow(2).sum() + selector_lambda * violation
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        weights = torch.softmax(logits / temperature, dim=0)
        update = weights @ candidates
        first = gradients @ update
        predicted = first
        if use_curvature:
            predicted = predicted + 0.5 * torch.einsum(
                "eij,i,j->e", hessians, update, update
            )
    diagnostics = {
        "selector_weight_entropy": float(
            -(weights * weights.clamp_min(1e-12).log()).sum()
        ),
        "predicted_first_max": float(first.max()),
        "predicted_quadratic_max": float(predicted.max()),
        "predicted_sign_reversals": float(((first < 0) & (predicted > 0)).sum()),
        "candidate_sign_reversals": float(
            ((candidate_first < 0) & (candidate_second > 0)).sum()
        ),
        "candidate_positive_quadratic": float((candidate_second > 0).sum()),
        "virtual_step_norm": float(update.norm()),
    }
    return update.detach(), diagnostics, first.detach(), predicted.detach()


def sample_split(
    source: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    generator: torch.Generator,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    inner, outer = [], []
    for x, y in source:
        permutation = torch.randperm(x.shape[0], generator=generator)
        count = min(batch_size, x.shape[0] // 2)
        inner.append((x[permutation[:count]], y[permutation[:count]]))
        outer.append((x[permutation[count : 2 * count]], y[permutation[count : 2 * count]]))
    return inner, outer


def sample_environment_split(
    source: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    generator: torch.Generator,
    holdout_fraction: float = 0.5,
    fixed_halves: bool = False,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Split source environments, then sample independently within each.

    Environment-level splitting is essential for the meta objective: the
    outer risk must represent a held-out environment mechanism, not another
    minibatch from the same environment seen by the selector.
    """
    count = len(source)
    holdout = max(1, min(count - 1, int(round(count * holdout_fraction))))
    if fixed_halves:
        split = count // 2
        inner_indices = list(range(split))
        outer_indices = list(range(split, count))
    else:
        permutation = torch.randperm(count, generator=generator)
        inner_indices = permutation[:-holdout].tolist()
        outer_indices = permutation[-holdout:].tolist()

    def sample(indices):
        batches = []
        for index in indices:
            x, y = source[index]
            take = min(batch_size, x.shape[0])
            order = torch.randperm(x.shape[0], generator=generator)[:take]
            batches.append((x[order], y[order]))
        return batches

    return sample(inner_indices), sample(outer_indices)


def step_erm_or_vrex(
    model: MLP,
    outer: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vrex_lambda: float,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    risks = torch.stack([F.binary_cross_entropy_with_logits(model(x), y) for x, y in outer])
    mean_risk = risks.mean()
    objective = mean_risk + vrex_lambda * (risks - mean_risk).pow(2).mean()
    objective.backward()
    optimizer.step()
    return {"outer_loss": float(mean_risk.detach()), "outer_response_penalty": 0.0}


def step_e2e(
    model: MLP,
    inner: list[tuple[torch.Tensor, torch.Tensor]],
    outer: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    method: str,
    alpha: float,
    selector_steps: int,
    selector_lr: float,
    selector_lambda: float,
    selector_temperature: float,
    outer_gamma: float,
    outer_objective: str,
) -> dict[str, float]:
    current_head = head_vector(model)
    inner_features = [model.feature(x).detach() for x, _ in inner]
    inner_labels = [y for _, y in inner]
    update, diagnostics, predicted_first, predicted_response = select_virtual_step(
        current_head,
        inner_features,
        inner_labels,
        alpha,
        use_curvature=method == "e2e_ccra",
        selector_steps=selector_steps,
        selector_lr=selector_lr,
        selector_lambda=selector_lambda,
        selector_temperature=selector_temperature,
    )
    perturbed_head = current_head + update
    optimizer.zero_grad(set_to_none=True)
    base_risks = torch.stack([custom_risk(model, x, y, current_head) for x, y in outer])
    perturbed_risks = torch.stack([custom_risk(model, x, y, perturbed_head) for x, y in outer])
    response = perturbed_risks - base_risks
    # Audit every candidate on the independent outer batch.  This is separate
    # from the selected update: it verifies that the source curvature witness
    # survives the finite step used by the outer objective.
    with torch.no_grad():
        outer_features = [model.feature(x) for x, _ in outer]
        outer_labels = [y for _, y in outer]
        outer_gradients, _ = head_geometry(current_head.detach(), outer_features, outer_labels)
        candidate_norms = outer_gradients.norm(dim=1, keepdim=True).clamp_min(1e-8)
        outer_candidates = -alpha * outer_gradients / candidate_norms
        candidate_first = outer_gradients @ outer_candidates.T
        candidate_exact = torch.stack(
            [
                torch.stack(
                    [
                        custom_risk(model, x, y, current_head + outer_candidates[k])
                        - custom_risk(model, x, y, current_head)
                        for x, y in outer
                    ]
                )
                for k in range(outer_candidates.shape[0])
            ],
            dim=1,
        )
    # Only harmful exact finite-step responses should shape the encoder.
    response_penalty = F.relu(response).pow(2).mean()
    if outer_objective == "meta":
        # Environment-level held-out risk is the transfer signal.  The
        # response penalty remains a diagnostic/constraint, not the target
        # surrogate that the old sample-split objective incorrectly used.
        objective = base_risks.mean() + outer_gamma * perturbed_risks.mean()
    else:
        objective = base_risks.mean() + outer_gamma * response_penalty
    objective.backward()
    optimizer.step()
    diagnostics.update(
        {
            "outer_loss": float(base_risks.mean().detach()),
            "outer_perturbed_loss": float(perturbed_risks.mean().detach()),
            "outer_response_penalty": float(response_penalty.detach()),
            "actual_response_mean": float(response.mean().detach()),
            "actual_response_max": float(response.max().detach()),
            "fo_sign_reversal_rate": float(
                ((predicted_first < 0) & (response.detach() > 0)).float().mean()
            ),
            "quadratic_prediction_error": float(
                (response.detach() - predicted_response).abs().mean()
            ),
            "candidate_exact_sign_reversals": float(
                ((candidate_first < 0) & (candidate_exact > 0)).sum()
            ),
            "candidate_exact_positive_responses": float((candidate_exact > 0).sum()),
        }
    )
    return diagnostics


def step_mldg_head(
    model: MLP,
    inner: list[tuple[torch.Tensor, torch.Tensor]],
    outer: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    alpha: float,
    outer_gamma: float,
) -> dict[str, float]:
    """A deliberately explicit head-only MLDG-style control.

    The first half of source environments proposes one head step; the second
    half evaluates its exact post-update loss.  It is included to separate the
    all-source response geometry of E2E-CCRA from ordinary train/test
    meta-splitting.
    """
    current_head = head_vector(model)
    inner_features = [model.feature(x).detach() for x, _ in inner]
    inner_labels = [y for _, y in inner]
    gradients, _ = head_geometry(current_head.detach(), inner_features, inner_labels)
    mean_gradient = gradients.mean(dim=0)
    update = -alpha * mean_gradient / mean_gradient.norm().clamp_min(1e-8)
    perturbed_head = current_head + update.detach()
    optimizer.zero_grad(set_to_none=True)
    base = torch.stack([custom_risk(model, x, y, current_head) for x, y in outer])
    perturbed = torch.stack([custom_risk(model, x, y, perturbed_head) for x, y in outer])
    response = perturbed - base
    penalty = F.relu(response).pow(2).mean()
    objective = base.mean() + outer_gamma * penalty
    objective.backward()
    optimizer.step()
    return {
        "outer_loss": float(base.mean().detach()),
        "outer_response_penalty": float(penalty.detach()),
        "actual_response_mean": float(response.mean().detach()),
        "actual_response_max": float(response.max().detach()),
        "fo_sign_reversal_rate": 0.0,
        "quadratic_prediction_error": 0.0,
        "virtual_step_norm": float(update.norm()),
    }


def run_method(
    method: str,
    base_state: dict[str, torch.Tensor],
    source: list[tuple[torch.Tensor, torch.Tensor]],
    target: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    batch_size: int,
    lr: float,
    alpha: float,
    selector_steps: int,
    selector_lr: float,
    selector_lambda: float,
    selector_temperature: float,
    outer_gamma: float,
    outer_objective: str,
    target_shift_mode: str,
    vrex_lambda: float,
    seed: int,
) -> dict[str, float]:
    model = CurvatureModel() if "fc1.weight" not in base_state else MLP()
    model.load_state_dict(copy.deepcopy(base_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 12000)
    history = []
    for _ in range(steps):
        if outer_objective == "meta" or target_shift_mode == "curvature":
            inner, outer = sample_environment_split(
                source,
                batch_size,
                generator,
                fixed_halves=target_shift_mode == "curvature",
            )
        else:
            inner, outer = sample_split(source, batch_size, generator)
        if method == "erm":
            info = step_erm_or_vrex(model, outer, optimizer, 0.0)
        elif method == "vrex":
            info = step_erm_or_vrex(model, outer, optimizer, vrex_lambda)
        elif method == "mldg_head":
            midpoint = len(inner) // 2
            info = step_mldg_head(
                model,
                inner[:midpoint],
                outer[midpoint:],
                optimizer,
                alpha,
                outer_gamma,
            )
        else:
            info = step_e2e(
                model,
                inner,
                outer,
                optimizer,
                method,
                alpha,
                selector_steps,
                selector_lr,
                selector_lambda,
                selector_temperature,
                outer_gamma,
                outer_objective,
            )
        history.append(info)
    source_eval = evaluate(
        model,
        (torch.cat([x for x, _ in source]), torch.cat([y for _, y in source])),
    )
    target_eval = evaluate(model, target)
    result = {
        "method": method,
        "source_loss": source_eval["loss"],
        "source_accuracy": source_eval["accuracy"],
        "target_loss": target_eval["loss"],
        "target_accuracy": target_eval["accuracy"],
    }
    for key in (
        "outer_response_penalty",
        "outer_perturbed_loss",
        "actual_response_mean",
        "actual_response_max",
        "predicted_sign_reversals",
        "candidate_sign_reversals",
        "candidate_positive_quadratic",
        "candidate_exact_sign_reversals",
        "candidate_exact_positive_responses",
        "fo_sign_reversal_rate",
        "quadratic_prediction_error",
        "predicted_quadratic_max",
        "virtual_step_norm",
    ):
        values = [entry[key] for entry in history if key in entry]
        if values:
            result["mean_" + key] = float(np.mean(values))
    predicted_reversals = result.get("mean_predicted_sign_reversals", 0.0)
    candidate_reversals = result.get("mean_candidate_sign_reversals", 0.0)
    exact_reversals = max(
        result.get("mean_fo_sign_reversal_rate", 0.0),
        result.get("mean_candidate_exact_sign_reversals", 0.0),
    )
    result["response_mechanism_active"] = bool(
        predicted_reversals > 0.0
        or candidate_reversals > 0.0
        or exact_reversals > 0.0
    )
    return result


def run_one(args, seed: int, gamma: float) -> dict[str, object]:
    set_seed(seed)
    if args.target_shift_mode == "curvature":
        source, target = make_curvature_datasets(args.n_per_env)
        warmup = CurvatureModel()
        with torch.no_grad():
            warmup.fc2.weight.zero_()
            warmup.fc2.bias.zero_()
    else:
        geometry = make_geometry(seed)
        source, _, target, _, _ = make_datasets(
            geometry,
            args.n_per_env,
            gamma,
            seed,
            target_shift_mode=args.target_shift_mode,
        )
        warmup = MLP()
    optimizer = torch.optim.Adam(warmup.parameters(), lr=args.lr)
    warmup_epochs = 0 if args.target_shift_mode == "curvature" else args.warmup_epochs
    for _ in range(warmup_epochs):
        optimizer.zero_grad(set_to_none=True)
        risks = torch.stack([F.binary_cross_entropy_with_logits(warmup(x), y) for x, y in source])
        risks.mean().backward()
        optimizer.step()
    base_state = copy.deepcopy(warmup.state_dict())
    rows = [
        run_method(
            method,
            base_state,
            source,
            target,
            args.steps,
            args.batch_size,
            args.lr,
            args.alpha,
            args.selector_steps,
            args.selector_lr,
            args.selector_lambda,
            args.selector_temperature,
            args.outer_gamma,
            args.outer_objective,
            args.target_shift_mode,
            args.vrex_lambda,
            seed,
        )
        for method in ("erm", "vrex", "mldg_head", "e2e_fo", "e2e_ccra")
    ]
    return {"seed": seed, "gamma": gamma, "target_shift_mode": args.target_shift_mode, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--selector-steps", type=int, default=12)
    parser.add_argument("--selector-lr", type=float, default=0.08)
    parser.add_argument("--selector-lambda", type=float, default=4.0)
    parser.add_argument("--selector-temperature", type=float, default=0.01)
    parser.add_argument("--outer-gamma", type=float, default=2.0)
    parser.add_argument(
        "--outer-objective", choices=["response", "meta"], default="response"
    )
    parser.add_argument("--vrex-lambda", type=float, default=1.0)
    parser.add_argument(
        "--target-shift-mode",
        choices=["predictive", "nuisance", "sign_reversal", "curvature"],
        default="predictive",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = [run_one(args, seed, gamma) for gamma in args.gammas for seed in args.seeds]
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
