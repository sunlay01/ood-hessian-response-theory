"""Regularizer-specific blind-transfer attack.

This is a theorem-validation experiment, not a new training algorithm.  It
keeps a small classifier head and computes the head transfer geometry directly:

    O = D_eta S_Omega,          G = D_eta (grad_phi R, rho/2 Hess_phi R).

For each trained source-only model, the script constructs blind and visible
mechanism shifts from ``G P_ker(O)`` and evaluates a Transfer-Measure-style
classifier-head attack on held-out data generated at those shifts.  The target
distribution and the latent eta coordinates are used only for this evaluation
oracle.  The source learner receives tensors and environment IDs only.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.nn import functional as F


torch.set_num_threads(1)

INPUT_DIM = 67
HIDDEN_DIM = 16
MECHANISM_DIM = 3
SOURCE_ENVS = 8
DEFAULT_N_PER_ENV = 512
DEFAULT_EVAL_N = 1024
DEFAULT_EPOCHS = 80
DEFAULT_LR = 3e-3
DEFAULT_REG_LAMBDA = 10.0
DEFAULT_FD_STEP = 2e-3
DEFAULT_SHIFT_EPS = 0.35
DEFAULT_ATTACK_RADIUS = 0.35
DEFAULT_ATTACK_STEPS = 30
DEFAULT_ATTACK_LR = 0.05


@dataclass
class Latents:
    image: np.ndarray
    y: np.ndarray
    y_pm: np.ndarray
    noise: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_eta_table() -> np.ndarray:
    """Eight centered source environments in a three-mechanism space."""

    return np.asarray(
        [
            [0.8, 0.1, 0.7],
            [-0.8, -0.1, -0.7],
            [0.1, 0.8, -0.6],
            [-0.1, -0.8, 0.6],
            [0.7, -0.7, 0.2],
            [-0.7, 0.7, -0.2],
            [0.4, 0.5, -0.8],
            [-0.4, -0.5, 0.8],
        ],
        dtype=float,
    )


def make_latents(n: int, seed: int) -> Latents:
    rng = np.random.default_rng(seed)
    digits = load_digits()
    indices = rng.integers(0, len(digits.data), size=n)
    image = digits.data[indices].astype(np.float32) / 16.0
    y = (digits.target[indices] >= 5).astype(np.float32)
    y_pm = 2.0 * y - 1.0
    noise = rng.normal(size=(n, MECHANISM_DIM)).astype(np.float32)
    return Latents(image=image, y=y, y_pm=y_pm, noise=noise)


def make_environment(latents: Latents, eta: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth three-mechanism augmentation of a fixed invariant digit sample."""

    eta = np.asarray(eta, dtype=np.float32)
    # color, texture and lighting/background proxies.  All are smooth in eta;
    # no target or latent eta is exposed to the learner.
    nuisance = latents.y_pm[:, None] * eta[None, :] + 0.85 * latents.noise
    x = np.concatenate([latents.image, nuisance], axis=1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(
        latents.y, dtype=torch.float32
    )


class DigitMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.feature(x)).squeeze(-1)


def irm_statistic(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scale = torch.ones((), dtype=logits.dtype, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(scale * logits, labels)
    return torch.autograd.grad(loss, scale, create_graph=True)[0]


def train_source_model(
    method: str,
    source: list[tuple[torch.Tensor, torch.Tensor]],
    base_state: dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    regularizer_lambda: float,
) -> DigitMLP:
    model = DigitMLP()
    model.load_state_dict(copy.deepcopy(base_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        irm_values = []
        for x, y in source:
            logits = model(x)
            losses.append(F.binary_cross_entropy_with_logits(logits, y))
            irm_values.append(irm_statistic(logits, y))
        mean_loss = torch.stack(losses).mean()
        if method == "erm":
            penalty = torch.zeros_like(mean_loss)
        elif method == "irm":
            penalty = torch.stack(irm_values).pow(2).mean()
        elif method == "vrex":
            penalty = torch.stack(losses).var(unbiased=False)
        else:
            raise ValueError(method)
        objective = mean_loss + regularizer_lambda * penalty
        objective.backward()
        optimizer.step()
    return model


def head_vector(model: DigitMLP) -> torch.Tensor:
    return torch.cat([model.fc2.weight.reshape(-1), model.fc2.bias.reshape(-1)])


def head_risk_from_vector(
    features: torch.Tensor,
    labels: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    logits = features @ vector[:-1] + vector[-1]
    return F.binary_cross_entropy_with_logits(logits, labels)


def head_geometry(
    model: DigitMLP,
    data: tuple[torch.Tensor, torch.Tensor],
    radius: float,
) -> np.ndarray:
    """Return [rho * head-gradient, rho^2/2 * flattened head-Hessian]."""

    x, y = data
    features = model.feature(x).detach()
    vector = head_vector(model).detach().requires_grad_(True)

    def risk(v: torch.Tensor) -> torch.Tensor:
        return head_risk_from_vector(features, y, v)

    gradient = torch.autograd.grad(risk(vector), vector, create_graph=False)[0]
    hessian = torch.autograd.functional.hessian(risk, vector)
    output = torch.cat(
        [
            radius * gradient.detach(),
            (radius**2 / 2.0) * hessian.reshape(-1).detach(),
        ]
    )
    return output.cpu().numpy()


def regularizer_statistic(
    model: DigitMLP,
    data: tuple[torch.Tensor, torch.Tensor],
    method: str,
) -> float:
    x, y = data
    features = model.feature(x).detach()
    vector = head_vector(model).detach().requires_grad_(True)
    logits = features @ vector[:-1] + vector[-1]
    if method == "irm":
        # IRMv1 response at the unit classifier scale.
        residual = torch.sigmoid(logits) - y
        return float(torch.mean(residual * logits).detach())
    if method in ("vrex", "erm"):
        return float(F.binary_cross_entropy_with_logits(logits, y).detach())
    raise ValueError(method)


def finite_difference_column(
    model: DigitMLP,
    latents: Latents,
    eta0: np.ndarray,
    direction: np.ndarray,
    step: float,
    geometry: bool,
    radius: float,
) -> np.ndarray:
    plus = make_environment(latents, eta0 + step * direction)
    minus = make_environment(latents, eta0 - step * direction)
    if geometry:
        plus_value = head_geometry(model, plus, radius)
        minus_value = head_geometry(model, minus, radius)
    else:
        plus_value = np.asarray(
            regularizer_statistic(model, plus, _CURRENT_METHOD), dtype=float
        )
        minus_value = np.asarray(
            regularizer_statistic(model, minus, _CURRENT_METHOD), dtype=float
        )
    return (plus_value - minus_value) / (2.0 * step)


_CURRENT_METHOD = "erm"


def observation_and_response(
    model: DigitMLP,
    latent: Latents,
    eta0: np.ndarray,
    method: str,
    fd_step: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    global _CURRENT_METHOD
    _CURRENT_METHOD = method
    columns = []
    for j in range(MECHANISM_DIM):
        direction = np.zeros(MECHANISM_DIM)
        direction[j] = 1.0
        columns.append(
            finite_difference_column(
                model,
                latent,
                eta0,
                direction,
                fd_step,
                geometry=False,
                radius=radius,
            )
        )
    observation = np.asarray(columns, dtype=float).reshape(1, MECHANISM_DIM)

    response_columns = []
    for j in range(MECHANISM_DIM):
        direction = np.zeros(MECHANISM_DIM)
        direction[j] = 1.0
        response_columns.append(
            finite_difference_column(
                model,
                latent,
                eta0,
                direction,
                fd_step,
                geometry=True,
                radius=radius,
            )
        )
    response = np.stack(response_columns, axis=1)
    projector = null_projector(observation)
    blind = response @ projector
    singular_values = np.linalg.svd(blind, compute_uv=False)
    kappa = float(np.linalg.svd(response @ np.linalg.pinv(observation), compute_uv=False)[0])
    diagnostics = {
        "beta": float(singular_values[0]) if len(singular_values) else 0.0,
        "response_norm": float(np.linalg.svd(response, compute_uv=False)[0]),
        "kappa": kappa,
        "observation_norm": float(np.linalg.norm(observation)),
        "blind_rank": float(np.linalg.matrix_rank(blind, tol=1e-8)),
    }
    return observation, response, diagnostics


def null_projector(observation: np.ndarray) -> np.ndarray:
    if observation.size == 0:
        return np.eye(MECHANISM_DIM)
    return np.eye(observation.shape[1]) - np.linalg.pinv(observation) @ observation


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(np.linalg.norm(vector), 1e-12)


def fit_head_optimum(features: torch.Tensor, labels: torch.Tensor, start: torch.Tensor) -> float:
    vector = start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vector], lr=0.08)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        risk = head_risk_from_vector(features, labels, vector)
        risk.backward()
        optimizer.step()
    return float(head_risk_from_vector(features, labels, vector).detach())


def transfer_attack(
    model: DigitMLP,
    source_data: tuple[torch.Tensor, torch.Tensor],
    target_data: tuple[torch.Tensor, torch.Tensor],
    radius: float,
    steps: int,
    attack_lr: float,
) -> dict[str, float]:
    source_features = model.feature(source_data[0]).detach()
    target_features = model.feature(target_data[0]).detach()
    source_labels, target_labels = source_data[1], target_data[1]
    start = head_vector(model).detach()
    source_opt = fit_head_optimum(source_features, source_labels, start)
    target_opt = fit_head_optimum(target_features, target_labels, start)

    def excess_gap(vector: torch.Tensor) -> torch.Tensor:
        target_excess = head_risk_from_vector(target_features, target_labels, vector) - target_opt
        source_excess = head_risk_from_vector(source_features, source_labels, vector) - source_opt
        return target_excess - source_excess

    values = []
    for sign in (-1.0, 1.0):
        delta = torch.zeros_like(start, requires_grad=True)
        for _ in range(steps):
            objective = sign * excess_gap(start + delta)
            gradient = torch.autograd.grad(objective, delta)[0]
            delta = (delta + attack_lr * gradient / gradient.norm().clamp_min(1e-12)).detach()
            delta = (delta * min(1.0, radius / delta.norm().clamp_min(1e-12))).requires_grad_(True)
        values.append(float(excess_gap(start + delta).detach()))
    return {
        "attack_gap": float(max(abs(value) for value in values)),
        "attack_signed_max": float(max(values)),
        "attack_signed_min": float(min(values)),
        "source_opt": source_opt,
        "target_opt": target_opt,
    }


def shift_metrics(
    model: DigitMLP,
    source_data: tuple[torch.Tensor, torch.Tensor],
    eval_latents: Latents,
    eta0: np.ndarray,
    xi: np.ndarray,
    observation: np.ndarray,
    response: np.ndarray,
    kappa: float,
    eps: float,
    radius: float,
    attack_steps: int,
    attack_lr: float,
    kind: str,
    index: int,
) -> dict[str, float | str | int]:
    projector = null_projector(observation)
    target_data = make_environment(eval_latents, eta0 + eps * xi)
    attack = transfer_attack(
        model, source_data, target_data, radius, attack_steps, attack_lr
    )
    visible = kappa * float(np.linalg.norm(observation @ xi))
    blind = float(np.linalg.norm(response @ projector @ xi))
    total = visible + blind
    return {
        "kind": kind,
        "index": index,
        "xi": [float(value) for value in xi],
        "visible": visible,
        "blind": blind,
        "certificate": total,
        "observation_abs": float(np.linalg.norm(observation @ xi)),
        "response_norm": float(np.linalg.norm(response @ xi)),
        "attack_gap": attack["attack_gap"],
        "attack_signed_max": attack["attack_signed_max"],
        "attack_signed_min": attack["attack_signed_min"],
    }


def choose_matched_pair(rows: list[dict]) -> dict[str, object] | None:
    random_rows = [row for row in rows if row["kind"] == "random"]
    if len(random_rows) < 2:
        return None
    observation_scale = max(
        max(float(row["observation_abs"]) for row in random_rows), 1e-12
    )
    blind_scale = max(max(float(row["blind"]) for row in random_rows), 1e-12)
    candidates = []
    for left, right in itertools.combinations(random_rows, 2):
        visible_gap = abs(float(left["observation_abs"]) - float(right["observation_abs"]))
        blind_gap = abs(float(left["blind"]) - float(right["blind"]))
        # Select the pair from theorem-side quantities only.  Outcome-dependent
        # pair selection would leak the transfer attack into the control.
        normalized_visible_gap = visible_gap / observation_scale
        normalized_blind_gap = blind_gap / blind_scale
        candidates.append(
            (
                normalized_visible_gap,
                normalized_blind_gap,
                visible_gap,
                blind_gap,
                left,
                right,
            )
        )
    eligible = [item for item in candidates if item[0] <= 0.10]
    if eligible:
        # Among shifts matched to within 10% of the observed visible range,
        # maximize the blind contrast before looking at attack outcomes.
        selected = max(eligible, key=lambda item: (item[1], -item[0]))
    else:
        # Deterministic fallback: best blind-separation per visible mismatch.
        selected = max(
            candidates,
            key=lambda item: (item[1] - item[0], item[1], -item[0]),
        )
    _, _, _, _, left, right = selected
    if left["blind"] < right["blind"]:
        left, right = right, left
    return {
        "high_blind": left,
        "low_blind": right,
        "visible_abs_difference": abs(
            float(left["observation_abs"]) - float(right["observation_abs"])
        ),
        "blind_difference": float(left["blind"]) - float(right["blind"]),
        "attack_difference": float(left["attack_gap"]) - float(right["attack_gap"]),
    }


def run_model(
    method: str,
    model: DigitMLP,
    source_latents: Latents,
    eval_latents: Latents,
    eta0: np.ndarray,
    fd_step: float,
    shift_eps: float,
    attack_radius: float,
    attack_steps: int,
    attack_lr: float,
    num_random_shifts: int,
    seed: int,
) -> dict[str, object]:
    observation, response, geometry_diag = observation_and_response(
        model, source_latents, eta0, method, fd_step, attack_radius
    )
    projector = null_projector(observation)
    blind_matrix = response @ projector
    _, _, right = np.linalg.svd(blind_matrix, full_matrices=False)
    blind_xi = normalized(right[0])
    visible_xi = normalized(observation.reshape(-1))
    # The response map is local at eta0, so the empirical transfer measure must
    # compare R_eta0 against R_{eta0 + eps xi}.  Reusing the same latent sample
    # on both sides isolates the mechanism shift from sampling noise.
    source_eval_data = make_environment(eval_latents, eta0)

    rows = [
        shift_metrics(
            model,
            source_eval_data,
            eval_latents,
            eta0,
            blind_xi,
            observation,
            response,
            geometry_diag["kappa"],
            shift_eps,
            attack_radius,
            attack_steps,
            attack_lr,
            "blind",
            0,
        ),
        shift_metrics(
            model,
            source_eval_data,
            eval_latents,
            eta0,
            visible_xi,
            observation,
            response,
            geometry_diag["kappa"],
            shift_eps,
            attack_radius,
            attack_steps,
            attack_lr,
            "visible",
            0,
        ),
    ]
    rng = np.random.default_rng(seed + 12345)
    for index in range(num_random_shifts):
        xi = normalized(rng.normal(size=MECHANISM_DIM))
        rows.append(
            shift_metrics(
                model,
                source_eval_data,
                eval_latents,
                eta0,
                xi,
                observation,
                response,
                geometry_diag["kappa"],
                shift_eps,
                attack_radius,
                attack_steps,
                attack_lr,
                "random",
                index,
            )
        )
    return {
        "method": method,
        "observation": observation.reshape(-1).tolist(),
        "response_shape": list(response.shape),
        "geometry": geometry_diag,
        "blind_xi": blind_xi.tolist(),
        "visible_xi": visible_xi.tolist(),
        "rows": rows,
        "matched_pair": choose_matched_pair(rows),
    }


def summarize(models: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for model in models:
        rows = model["rows"]
        random_rows = [row for row in rows if row["kind"] == "random"]
        result[model["method"]] = {
            "blind_attack_gap": next(row["attack_gap"] for row in rows if row["kind"] == "blind"),
            "visible_attack_gap": next(row["attack_gap"] for row in rows if row["kind"] == "visible"),
            "random_attack_gap_mean": float(np.mean([row["attack_gap"] for row in random_rows])),
            "random_certificate_corr": float(
                np.corrcoef(
                    [row["certificate"] for row in random_rows],
                    [row["attack_gap"] for row in random_rows],
                )[0, 1]
            ),
            "random_visible_corr": float(
                np.corrcoef(
                    [row["visible"] for row in random_rows],
                    [row["attack_gap"] for row in random_rows],
                )[0, 1]
            ),
            "random_blind_corr": float(
                np.corrcoef(
                    [row["blind"] for row in random_rows],
                    [row["attack_gap"] for row in random_rows],
                )[0, 1]
            ),
            "random_correlation_gain": float(
                np.corrcoef(
                    [row["certificate"] for row in random_rows],
                    [row["attack_gap"] for row in random_rows],
                )[0, 1]
                - np.corrcoef(
                    [row["visible"] for row in random_rows],
                    [row["attack_gap"] for row in random_rows],
                )[0, 1]
            ),
            "matched_pair": model["matched_pair"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-env", type=int, default=DEFAULT_N_PER_ENV)
    parser.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--regularizer-lambda", type=float, default=DEFAULT_REG_LAMBDA)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--shift-eps", type=float, default=DEFAULT_SHIFT_EPS)
    parser.add_argument("--attack-radius", type=float, default=DEFAULT_ATTACK_RADIUS)
    parser.add_argument("--attack-steps", type=int, default=DEFAULT_ATTACK_STEPS)
    parser.add_argument("--attack-lr", type=float, default=DEFAULT_ATTACK_LR)
    parser.add_argument("--random-shifts", type=int, default=24)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    eta0 = np.zeros(MECHANISM_DIM, dtype=float)
    source_etas = source_eta_table()
    source_latents = make_latents(args.n_per_env, args.seed + 10)
    eval_latents = make_latents(args.eval_n, args.seed + 20)
    source = [make_environment(source_latents, eta) for eta in source_etas]
    initial = DigitMLP()
    base_state = copy.deepcopy(initial.state_dict())
    models = []
    for method in ("erm", "irm", "vrex"):
        trained = train_source_model(
            method,
            source,
            base_state,
            args.epochs,
            args.lr,
            args.regularizer_lambda,
        )
        models.append(
            run_model(
                method,
                trained,
                source_latents,
                eval_latents,
                eta0,
                args.fd_step,
                args.shift_eps,
                args.attack_radius,
                args.attack_steps,
                args.attack_lr,
                args.random_shifts,
                args.seed,
            )
        )
    output = {
        "configuration": {
            "seed": args.seed,
            "n_per_env": args.n_per_env,
            "eval_n": args.eval_n,
            "source_envs": SOURCE_ENVS,
            "mechanism_dim": MECHANISM_DIM,
            "source_eta_table": source_etas.tolist(),
            "eta0": eta0.tolist(),
            "epochs": args.epochs,
            "lr": args.lr,
            "regularizer_lambda": args.regularizer_lambda,
            "fd_step": args.fd_step,
            "shift_eps": args.shift_eps,
            "attack_radius": args.attack_radius,
            "attack_steps": args.attack_steps,
            "attack_lr": args.attack_lr,
            "random_shifts": args.random_shifts,
            "target_used_for_training": False,
            "eta_used_for_training": False,
            "geometry_scope": "classifier_head_gradient_and_hessian",
        },
        "models": models,
        "summary": summarize(models),
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
