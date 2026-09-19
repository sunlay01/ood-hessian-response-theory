"""Hidden-environment finite-sample TSR stress test.

The generator has latent environment coefficients, but the learner receives
only source tensors and environment IDs.  Candidate statistics are generated
from the current hidden representation (environment means and projected
variances), and the source transfer map is estimated from gradient/HVP
sketches in the environment contrast space.

This is intentionally a controlled classification experiment rather than a
DomainBed implementation.  It tests four concrete failure modes:

* finite-sample response estimation;
* incomplete source coverage of the latent spurious family;
* automatically generated, rather than hand-written, statistic candidates;
* static versus periodically refreshed statistic selection.

The default command is a small smoke run.  The paper-facing grid can be run
with ``--sample-sizes 256 1024 4096 --gammas 0 0.5 1 --seeds 0 1 2 3 4``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


torch.set_num_threads(1)

INPUT_DIM = 20
HIDDEN_DIM = 32
SPURIOUS_DIM = 5
SOURCE_ENVS = 8
SPURIOUS_TANGENT_DIM = 3
TOP_K = 3
RANK_BUDGET = 2
DEFAULT_EPOCHS = 24
DEFAULT_WARMUP_EPOCHS = 6
DEFAULT_REFRESH_EVERY = 5
DEFAULT_LR = 2e-3
DEFAULT_IRM_LAMBDA = 10.0
DEFAULT_STAT_LAMBDA = 2.0
DEFAULT_MG = 8
DEFAULT_MH = 4
CERT_LAMBDA = 0.05


@dataclass
class Geometry:
    mixing: np.ndarray
    tangent_basis: np.ndarray
    source_coordinates: np.ndarray
    target_parallel_coordinate: np.ndarray
    target_perp: np.ndarray
    source_a: np.ndarray


@dataclass
class CandidateSpec:
    name: str
    kind: str
    direction: torch.Tensor
    scale: float
    values: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def contrast_matrix(num_envs: int) -> np.ndarray:
    projector = np.eye(num_envs) - np.ones((num_envs, num_envs)) / num_envs
    left, _, _ = np.linalg.svd(projector)
    return left[:, : num_envs - 1]


def make_geometry(seed: int) -> Geometry:
    rng = np.random.default_rng(seed + 101)
    mixing = rng.normal(size=(INPUT_DIM, 1 + SPURIOUS_DIM))
    mixing /= np.linalg.norm(mixing, axis=0, keepdims=True)
    basis, _ = np.linalg.qr(rng.normal(size=(SPURIOUS_DIM, SPURIOUS_DIM)))
    tangent_basis = basis[:, :SPURIOUS_TANGENT_DIM]
    target_perp = basis[:, SPURIOUS_TANGENT_DIM]
    source_coordinates = rng.normal(
        loc=0.0, scale=0.9, size=(SOURCE_ENVS, SPURIOUS_TANGENT_DIM)
    )
    # The parallel target coordinate is source-compatible but not equal to a
    # source environment, so gamma=0 is not a trivial train/test duplicate.
    target_parallel_coordinate = np.array([0.25, -0.30, 0.15])
    source_a = source_coordinates @ tangent_basis.T
    return Geometry(
        mixing=mixing,
        tangent_basis=tangent_basis,
        source_coordinates=source_coordinates,
        target_parallel_coordinate=target_parallel_coordinate,
        target_perp=target_perp,
        source_a=source_a,
    )


def sample_environment(
    geometry: Geometry,
    a: np.ndarray,
    n: int,
    rng: np.random.Generator,
    spurious_offset: np.ndarray | None = None,
    spurious_extra_noise: np.ndarray | None = None,
    sigma_invariant: float = 0.35,
    sigma_spurious: float = 0.80,
    sigma_x: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_pm = rng.choice(np.array([-1.0, 1.0]), size=n)
    z_i = y_pm + sigma_invariant * rng.normal(size=n)
    offset = np.zeros(SPURIOUS_DIM) if spurious_offset is None else spurious_offset
    z_s = a[None, :] * y_pm[:, None] + offset[None, :] + sigma_spurious * rng.normal(
        size=(n, SPURIOUS_DIM)
    )
    if spurious_extra_noise is not None:
        z_s += rng.normal(size=(n, 1)) * spurious_extra_noise[None, :]
    z = np.concatenate([z_i[:, None], z_s], axis=1)
    x = np.tanh(z @ geometry.mixing.T)
    x += sigma_x * rng.normal(size=x.shape)
    y = (y_pm + 1.0) / 2.0
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def make_datasets(
    geometry: Geometry,
    n_per_env: int,
    gamma: float,
    seed: int,
    estimate_fraction: float = 0.5,
    target_n: int = 2048,
    target_shift_mode: str = "predictive",
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    list[tuple[torch.Tensor, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor],
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed + 1001)
    source = [
        sample_environment(geometry, a, n_per_env, rng)
        for a in geometry.source_a
    ]
    estimate_source = []
    n_est = max(32, int(n_per_env * estimate_fraction))
    for x, y in source:
        estimate_source.append((x[:n_est].clone(), y[:n_est].clone()))
    target_parallel = geometry.tangent_basis @ geometry.target_parallel_coordinate
    if target_shift_mode == "predictive":
        target_a = target_parallel + gamma * geometry.target_perp
        target_offset = np.zeros(SPURIOUS_DIM)
        target_extra_noise = None
    elif target_shift_mode == "nuisance":
        target_a = target_parallel
        target_offset = np.zeros(SPURIOUS_DIM)
        target_extra_noise = gamma * geometry.target_perp
    else:
        raise ValueError(f"unknown target shift mode: {target_shift_mode}")
    target = sample_environment(
        geometry,
        target_a,
        target_n,
        rng,
        spurious_offset=target_offset,
        spurious_extra_noise=target_extra_noise,
    )
    return source, estimate_source, target, target_a, target_offset


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.feature(x)).squeeze(-1)


def parameters_tuple(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(p for p in model.parameters() if p.requires_grad)


def flatten_tensors(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def make_probes(model: nn.Module, mg: int, mh: int, seed: int) -> tuple[torch.Tensor, ...]:
    dim = sum(p.numel() for p in model.parameters())
    generator = torch.Generator().manual_seed(seed + 7001)
    probes = []
    for _ in range(mg + mh):
        probe = torch.randn(dim, generator=generator)
        probe /= probe.norm().clamp_min(1e-12)
        probes.append(probe)
    return tuple(probes)


def irm_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    scale = torch.ones((), dtype=logits.dtype, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(scale * logits, labels)
    return torch.autograd.grad(loss, scale, create_graph=create_graph)[0]


def irm_statistics_nograd(logits: torch.Tensor, labels: torch.Tensor) -> float:
    residual = torch.sigmoid(logits) - labels
    return float(torch.mean(residual * logits).detach())


def candidate_value_from_feature(
    feature: torch.Tensor,
    spec_kind: str,
    direction: torch.Tensor,
) -> torch.Tensor:
    projection = feature @ direction
    if spec_kind == "mean":
        return projection.mean()
    if spec_kind == "variance":
        return ((projection - projection.mean()) ** 2).mean()
    raise ValueError(f"unknown candidate kind: {spec_kind}")


def candidate_values(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    kind: str,
    direction: torch.Tensor,
) -> torch.Tensor:
    values = []
    for x, _ in envs:
        feature = model.feature(x)
        values.append(candidate_value_from_feature(feature, kind, direction))
    return torch.stack(values)


def candidate_bank(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    top_k: int = TOP_K,
) -> list[CandidateSpec]:
    with torch.no_grad():
        means = []
        for x, _ in envs:
            means.append(model.feature(x).mean(dim=0))
        means_matrix = torch.stack(means, dim=0)
        centered = (means_matrix - means_matrix.mean(dim=0)).T
        left, _, _ = torch.linalg.svd(centered, full_matrices=False)
        directions = left[:, : min(top_k, left.shape[1])]
    bank: list[CandidateSpec] = []
    for k in range(directions.shape[1]):
        direction = directions[:, k].detach()
        for kind in ("mean", "variance"):
            values = candidate_values(model, envs, kind, direction).detach().cpu().numpy()
            centered_values = values - values.mean()
            scale = float(np.linalg.norm(centered_values @ contrast_matrix(len(envs))))
            bank.append(
                CandidateSpec(
                    name=f"{kind}_{k}",
                    kind=kind,
                    direction=direction.clone(),
                    scale=max(scale, 1e-5),
                    values=values,
                )
            )
    return bank


def baseline_row(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
) -> np.ndarray:
    values = []
    with torch.no_grad():
        for x, y in envs:
            values.append(irm_statistics_nograd(model(x), y))
    row = np.asarray(values) @ contrasts
    norm = np.linalg.norm(row)
    return row / max(norm, 1e-8)


def candidate_row(spec: CandidateSpec, contrasts: np.ndarray) -> np.ndarray:
    row = spec.values @ contrasts
    norm = np.linalg.norm(row)
    return row / max(norm, 1e-8)


def null_projector(observation: np.ndarray) -> np.ndarray:
    if observation.shape[0] == 0:
        return np.eye(observation.shape[1])
    projector = np.linalg.pinv(observation, rcond=1e-8) @ observation
    projector = (projector + projector.T) / 2.0
    return np.eye(observation.shape[1]) - projector


def op_norm(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def certificate(g_hat: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    projector = null_projector(observation)
    beta = op_norm(g_hat @ projector)
    kappa = op_norm(g_hat @ np.linalg.pinv(observation, rcond=1e-8))
    scale = op_norm(observation)
    return {
        "beta": beta,
        "kappa": kappa,
        "j_cert": beta + CERT_LAMBDA * kappa * scale,
    }


def transfer_sketch(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    probes: tuple[torch.Tensor, ...],
    mode: str,
) -> np.ndarray:
    params = parameters_tuple(model)
    mg = DEFAULT_MG
    mh = 0 if mode == "gradient" else DEFAULT_MH
    columns = []
    was_training = model.training
    model.eval()
    for x, y in envs:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        need_graph = mh > 0
        gradient = torch.autograd.grad(
            loss, params, create_graph=need_graph, retain_graph=need_graph
        )
        flat_gradient = flatten_tensors(gradient)
        sketch = [float(torch.dot(flat_gradient, probes[j]).detach()) for j in range(mg)]
        if mh:
            for j in range(mg, mg + mh):
                directional = torch.dot(flat_gradient, probes[j])
                hvp = torch.autograd.grad(
                    directional,
                    params,
                    retain_graph=(j < mg + mh - 1),
                    allow_unused=True,
                )
                flat_hvp = flatten_tensors(
                    value if value is not None else torch.zeros_like(param)
                    for value, param in zip(hvp, params)
                )
                sketch.append(float(torch.dot(flat_hvp, probes[j]).detach()))
        columns.append(sketch)
    model.train(was_training)
    return np.asarray(columns, dtype=float).T


def finite_response_error(g_hat: np.ndarray, g_reference: np.ndarray) -> float:
    denominator = max(op_norm(g_reference), 1e-8)
    return op_norm(g_hat - g_reference) / denominator


def select_indices(
    g_hat: np.ndarray,
    o0: np.ndarray,
    bank: list[CandidateSpec],
    contrasts: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    rank_budget: int = RANK_BUDGET,
) -> tuple[list[int], dict]:
    rows = [candidate_row(spec, contrasts) for spec in bank]
    if strategy == "random":
        selected = sorted(rng.choice(len(bank), size=rank_budget, replace=False).tolist())
    elif strategy == "raw_magnitude":
        scores = [op_norm(g_hat @ row[:, None]) for row in rows]
        selected = sorted(np.argsort(scores)[-rank_budget:].tolist())
    elif strategy == "tsr":
        candidates = []
        for subset in itertools.combinations(range(len(bank)), rank_budget):
            observation = np.vstack([o0] + [rows[i] for i in subset])
            score = certificate(g_hat, observation)["j_cert"]
            candidates.append((score, subset))
        _, best = min(candidates, key=lambda item: item[0])
        selected = list(best)
    else:
        raise ValueError(strategy)
    observation = np.vstack([o0] + [rows[i] for i in selected])
    return selected, certificate(g_hat, observation)


def objective(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    selected: list[CandidateSpec],
    irm_lambda: float,
    stat_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    irm_values = []
    candidate_values_by_spec = [[] for _ in selected]
    for x, y in envs:
        feature = model.feature(x)
        logits = model.fc2(feature).squeeze(-1)
        losses.append(F.binary_cross_entropy_with_logits(logits, y))
        irm_values.append(irm_statistics(logits, y, create_graph=True))
        for index, spec in enumerate(selected):
            candidate_values_by_spec[index].append(
                candidate_value_from_feature(feature, spec.kind, spec.direction)
            )
    source_loss = torch.stack(losses).mean()
    irm_penalty = torch.stack(irm_values).pow(2).mean()
    stat_penalty = torch.zeros_like(source_loss)
    for spec, values in zip(selected, candidate_values_by_spec):
        values_tensor = torch.stack(values) / max(spec.scale, 1e-5)
        stat_penalty = stat_penalty + (values_tensor - values_tensor.mean()).pow(2).mean()
    total = source_loss + irm_lambda * irm_penalty + stat_lambda * stat_penalty
    return total, {
        "source_loss": float(source_loss.detach()),
        "irm_penalty": float(irm_penalty.detach()),
        "stat_penalty": float(stat_penalty.detach()),
    }


def train_epochs(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    selected: list[CandidateSpec],
    epochs: int,
    lr: float,
    irm_lambda: float,
    stat_lambda: float,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = {}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(model, envs, selected, irm_lambda, stat_lambda)
        loss.backward()
        optimizer.step()
    return metrics


def evaluate(model: MLP, data: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x, y = data
        logits = model(x)
        loss = float(F.binary_cross_entropy_with_logits(logits, y))
        accuracy = float(((logits >= 0).float() == y).float().mean())
    return {"loss": loss, "accuracy": accuracy}


def clone_model(state: dict[str, torch.Tensor]) -> MLP:
    model = MLP()
    model.load_state_dict(copy.deepcopy(state))
    return model


def select_at_state(
    model: MLP,
    envs: list[tuple[torch.Tensor, torch.Tensor]],
    estimate_envs: list[tuple[torch.Tensor, torch.Tensor]],
    contrasts: np.ndarray,
    probes: tuple[torch.Tensor, ...],
    mode: str,
    rng: np.random.Generator,
    include_reference: bool = False,
) -> dict:
    bank = candidate_bank(model, envs)
    o0 = baseline_row(model, envs, contrasts)
    transfer_hat = transfer_sketch(model, estimate_envs, probes, mode)
    g_hat = transfer_hat @ contrasts
    if include_reference:
        transfer_reference = transfer_sketch(model, envs, probes, mode)
        g_reference = transfer_reference @ contrasts
    else:
        g_reference = g_hat
    selections = {}
    diagnostics = {}
    for strategy in ("random", "raw_magnitude", "tsr"):
        selected, cert = select_indices(
            g_hat, o0, bank, contrasts, strategy, rng
        )
        selections[strategy] = selected
        diagnostics[strategy] = cert
    reference_selected, reference_cert = select_indices(
        g_reference, o0, bank, contrasts, "tsr", rng
    )
    singular_values = np.linalg.svd(
        g_reference @ null_projector(o0[None, :]), compute_uv=False
    )
    gap = float(
        singular_values[RANK_BUDGET - 1] - singular_values[RANK_BUDGET]
        if len(singular_values) > RANK_BUDGET
        else singular_values[-1]
    )
    return {
        "bank": bank,
        "o0": o0,
        "g_hat": g_hat,
        "g_reference": g_reference,
        "transfer_reference": transfer_reference if include_reference else None,
        "selections": selections,
        "diagnostics": diagnostics,
        "reference_selected": reference_selected,
        "reference_cert": reference_cert,
        "reference_gap": max(gap, 0.0),
        "estimation_error_abs": op_norm(g_hat - g_reference),
        "estimation_error_relative": finite_response_error(g_hat, g_reference),
    }


def selected_specs(bank: list[CandidateSpec], indices: list[int]) -> list[CandidateSpec]:
    return [bank[index] for index in indices]


def method_run(
    method: str,
    base_state: dict[str, torch.Tensor],
    source: list[tuple[torch.Tensor, torch.Tensor]],
    estimate_source: list[tuple[torch.Tensor, torch.Tensor]],
    target: tuple[torch.Tensor, torch.Tensor],
    contrasts: np.ndarray,
    probes: tuple[torch.Tensor, ...],
    mode: str,
    initial_selection: dict,
    seed: int,
    epochs: int,
    warmup_epochs: int,
    refresh_every: int,
    lr: float,
    irm_lambda: float,
    stat_lambda: float,
) -> dict:
    model = clone_model(base_state)
    bank = initial_selection["bank"]
    selections = initial_selection["selections"]
    if method == "irm":
        selected_indices = []
    else:
        selection_key = "tsr" if method in ("tsr_static", "tsr_refresh") else method
        selected_indices = selections[selection_key]
    turnover = 0
    refresh_count = 0
    last_indices = list(selected_indices)
    selected = selected_specs(bank, selected_indices)
    remaining = max(0, epochs - warmup_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(remaining):
        if method == "tsr_refresh" and epoch % refresh_every == 0:
            current = select_at_state(
                model,
                source,
                estimate_source,
                contrasts,
                probes,
                mode,
                np.random.default_rng(seed + 8000 + epoch),
                include_reference=False,
            )
            bank = current["bank"]
            new_indices = current["selections"]["tsr"]
            turnover += int(new_indices != last_indices)
            last_indices = list(new_indices)
            selected = selected_specs(bank, new_indices)
            refresh_count += 1
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(
            model,
            source,
            selected,
            irm_lambda,
            stat_lambda,
        )
        loss.backward()
        optimizer.step()
        history.append(metrics)
    source_metric = evaluate(model, (torch.cat([x for x, _ in source]), torch.cat([y for _, y in source])))
    target_metric = evaluate(model, target)
    final_selection = select_at_state(
        model,
        source,
        estimate_source,
        contrasts,
        probes,
        mode,
        np.random.default_rng(seed + 9000),
        include_reference=False,
    )
    return {
        "method": method,
        "source_loss": source_metric["loss"],
        "source_accuracy": source_metric["accuracy"],
        "target_loss": target_metric["loss"],
        "target_accuracy": target_metric["accuracy"],
        "selected_names": [spec.name for spec in selected],
        "selected_indices": selected_indices,
        "final_selected_names": [
            final_selection["bank"][i].name for i in final_selection["selections"]["tsr"]
        ],
        "selection_turnover": turnover,
        "refresh_count": refresh_count,
        "epochs_after_warmup": remaining,
        "history_last": history[-1] if history else {},
    }


def run_setting(
    n_per_env: int,
    gamma: float,
    seed: int,
    mode: str,
    target_shift_mode: str,
    epochs: int,
    warmup_epochs: int,
    refresh_every: int,
    lr: float,
    irm_lambda: float,
    stat_lambda: float,
) -> dict:
    set_seed(seed)
    geometry = make_geometry(seed)
    source, estimate_source, target, target_a, target_offset = make_datasets(
        geometry, n_per_env, gamma, seed, target_shift_mode=target_shift_mode
    )
    contrasts = contrast_matrix(SOURCE_ENVS)
    warmup = MLP()
    train_epochs(warmup, source, [], warmup_epochs, lr, irm_lambda, 0.0)
    base_state = copy.deepcopy(warmup.state_dict())
    probes = make_probes(warmup, DEFAULT_MG, DEFAULT_MH, seed)
    initial = select_at_state(
        warmup,
        source,
        estimate_source,
        contrasts,
        probes,
        mode,
        np.random.default_rng(seed + 6000),
        include_reference=True,
    )
    methods = ["irm", "random", "raw_magnitude", "tsr_static", "tsr_refresh"]
    rows = [
        method_run(
            method,
            base_state,
            source,
            estimate_source,
            target,
            contrasts,
            probes,
            mode,
            initial,
            seed,
            epochs,
            warmup_epochs,
            refresh_every,
            lr,
            irm_lambda,
            stat_lambda,
        )
        for method in methods
    ]
    static_tsr = initial["selections"]["tsr"]
    reference_tsr = initial["reference_selected"]
    overlap = len(set(static_tsr) & set(reference_tsr)) / RANK_BUDGET
    target_transfer = transfer_sketch(warmup, [target], probes, mode)[:, 0]
    source_transfer = initial["transfer_reference"]
    source_projector = source_transfer @ np.linalg.pinv(source_transfer, rcond=1e-8)
    coverage_residual = (np.eye(source_transfer.shape[0]) - source_projector) @ target_transfer
    coverage_defect_abs = float(np.linalg.norm(coverage_residual))
    phase_ratio = (
        initial["estimation_error_abs"] + coverage_defect_abs
    ) / max(initial["reference_gap"], 1e-8)
    return {
        "n_per_env": n_per_env,
        "gamma": gamma,
        "target_shift_mode": target_shift_mode,
        "seed": seed,
        "sketch_mode": mode,
        "target_a_norm": float(np.linalg.norm(target_a)),
        "target_offset_norm": float(np.linalg.norm(target_offset)),
        # These are evaluation-only diagnostics.  The learner never receives
        # target_transfer, target_a, or the latent gamma.
        "coverage_defect_latent": float(abs(gamma)),
        "coverage_defect_transfer_abs": coverage_defect_abs,
        "proxy_error_gap_ratio": phase_ratio,
        "initial_selection": {
            "candidate_names": [spec.name for spec in initial["bank"]],
            "tsr_names": [initial["bank"][i].name for i in static_tsr],
            "reference_tsr_names": [initial["bank"][i].name for i in reference_tsr],
            "raw_names": [initial["bank"][i].name for i in initial["selections"]["raw_magnitude"]],
            "random_names": [initial["bank"][i].name for i in initial["selections"]["random"]],
            "tsr_reference_overlap": overlap,
            "reference_gap": initial["reference_gap"],
            "estimation_error_abs": initial["estimation_error_abs"],
            "estimation_error_relative": initial["estimation_error_relative"],
            "selection_cert": initial["diagnostics"]["tsr"],
        },
        "rows": rows,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = [
        row
        | {
            k: rec[k]
            for k in ("n_per_env", "gamma", "seed", "sketch_mode", "target_shift_mode")
        }
        for rec in records
        for row in rec["rows"]
    ]
    summary = []
    for key, group in itertools.groupby(
        sorted(
            rows,
            key=lambda r: (
                r["target_shift_mode"],
                r["sketch_mode"],
                r["n_per_env"],
                r["gamma"],
                r["method"],
            ),
        ),
        key=lambda r: (
            r["target_shift_mode"],
            r["sketch_mode"],
            r["n_per_env"],
            r["gamma"],
            r["method"],
        ),
    ):
        group = list(group)
        item = {
            "target_shift_mode": key[0],
            "sketch_mode": key[1],
            "n_per_env": key[2],
            "gamma": key[3],
            "method": key[4],
        }
        for metric in ("target_loss", "target_accuracy", "source_accuracy", "selection_turnover"):
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
        choices=["predictive", "nuisance"],
        default=["predictive"],
    )
    parser.add_argument("--sketch-modes", nargs="+", choices=["gradient", "gradient_hvp"], default=["gradient_hvp"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--refresh-every", type=int, default=DEFAULT_REFRESH_EVERY)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--irm-lambda", type=float, default=DEFAULT_IRM_LAMBDA)
    parser.add_argument("--stat-lambda", type=float, default=DEFAULT_STAT_LAMBDA)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = []
    for target_shift_mode in args.target_shift_modes:
        for mode in args.sketch_modes:
            for n_per_env in args.sample_sizes:
                for gamma in args.gammas:
                    for seed in args.seeds:
                        records.append(
                            run_setting(
                                n_per_env,
                                gamma,
                                seed,
                                mode,
                                target_shift_mode,
                                args.epochs,
                                args.warmup_epochs,
                                args.refresh_every,
                                args.lr,
                                args.irm_lambda,
                                args.stat_lambda,
                            )
                        )
    output = {
        "configuration": {
            "source_envs": SOURCE_ENVS,
            "spurious_dim": SPURIOUS_DIM,
            "spurious_tangent_dim": SPURIOUS_TANGENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "rank_budget": RANK_BUDGET,
            "top_k": TOP_K,
            "gradient_probes": DEFAULT_MG,
            "hvp_probes": DEFAULT_MH,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "refresh_every": args.refresh_every,
            "sample_sizes": args.sample_sizes,
            "gammas": args.gammas,
            "seeds": args.seeds,
            "sketch_modes": args.sketch_modes,
            "target_shift_modes": args.target_shift_modes,
            "learning_rate": args.lr,
            "irm_lambda": args.irm_lambda,
            "stat_lambda": args.stat_lambda,
            "estimate_fraction": 0.5,
            "candidate_families": ["hidden_mean", "hidden_projected_variance"],
            "algorithm_latent_environment_access": False,
            "target_data_used_for_selection": False,
            "coverage_diagnostic_is_evaluation_only": True,
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
