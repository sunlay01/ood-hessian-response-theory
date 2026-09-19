"""Discriminative TSR probe: non-diagonal response and conditioning trap.

The candidate with the largest raw response is deliberately almost redundant
with the baseline observation.  A second candidate has a smaller raw response
but removes the dominant remaining blind direction.  The experiment compares
naive raw-response ranking with the scale-aware TSR certificate:

    J_cert(A) = beta(A) + lambda_cert * kappa(A) * ||A||_op.

The risk is nonlinear in rotated coordinates, q=5, rank(O0)=1, and the
candidate rows are not coordinate axes in the ambient parameter basis.  The
script jointly retrains the predictor after each selection with matched
statistic-gradient budget, and reports the final (theta-dependent) observation
rows as well as the normalized selection-time diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


Q = 5
_rng_basis = np.random.default_rng(123)
V, _ = np.linalg.qr(_rng_basis.normal(size=(Q, Q)))
SIGMA = np.diag([10.0, 3.0, 4.0, 1.0, 0.5])
B = V @ SIGMA @ V.T
AINV = V[:, 0]
ETA_BAR = 0.8 * V[:, 2]
SOURCE_DELTA = 0.30
BASE_LAMBDA = 1.0
CANDIDATE_LAMBDA = 20.0
CERT_LAMBDA = 0.05
TOTAL_STATISTIC_GRAD_BUDGET = 6000
TRAP_EPS = 0.04

V0, V1, V2, V3, V4 = [V[:, j] for j in range(Q)]
TRAP_ROW = (V0 + TRAP_EPS * V1) / np.linalg.norm(V0 + TRAP_EPS * V1)
CANDIDATE_ROWS = [TRAP_ROW, V2, V3, V4]
CANDIDATE_NAMES = ["near_visible_trap", "blind_dominant", "third", "fourth"]


def source_environments() -> np.ndarray:
    envs = [ETA_BAR.copy()]
    for j in range(Q):
        e = np.zeros(Q)
        e[j] = SOURCE_DELTA
        envs.extend([ETA_BAR + e, ETA_BAR - e])
    return np.asarray(envs)


SOURCE_ETAS = source_environments()


def log_cosh(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, -x) - np.log(2.0)


def alpha_eta(eta: np.ndarray) -> float:
    return 0.25 + 0.05 * np.tanh(float(V0 @ eta))


def target_center(eta: np.ndarray) -> np.ndarray:
    return AINV + B @ eta


def risk(theta: np.ndarray, eta: np.ndarray) -> float:
    rotated = V.T @ theta
    residual = theta - target_center(eta)
    return float(
        0.5 * np.dot(residual, residual)
        + alpha_eta(eta) * np.sum(log_cosh(rotated))
    )


def risk_grad(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    rotated = V.T @ theta
    residual = theta - target_center(eta)
    return residual + alpha_eta(eta) * (V @ np.tanh(rotated))


def risk_hessian(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    rotated = V.T @ theta
    diagonal = 1.0 + alpha_eta(eta) * (1.0 - np.tanh(rotated) ** 2)
    return V @ np.diag(diagonal) @ V.T


def transfer_vector(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [risk_grad(theta, eta), 0.5 * risk_hessian(theta, eta).reshape(-1)]
    )


def finite_response(theta: np.ndarray, eta0: np.ndarray, step: float) -> np.ndarray:
    cols = []
    for j in range(Q):
        e = np.zeros(Q)
        e[j] = 1.0
        cols.append(
            (transfer_vector(theta, eta0 + step * e)
             - transfer_vector(theta, eta0 - step * e))
            / (2.0 * step)
        )
    return np.column_stack(cols)


def op_norm(a: np.ndarray) -> float:
    if a.size == 0 or min(a.shape) == 0:
        return 0.0
    return float(np.linalg.svd(a, compute_uv=False)[0])


def null_projector(a: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0:
        return np.eye(Q)
    p = np.linalg.pinv(a, rcond=1e-10) @ a
    p = (p + p.T) / 2.0
    q = np.eye(Q) - p
    return (q + q.T) / 2.0


def beta_value(g: np.ndarray, o: np.ndarray) -> float:
    return op_norm(g @ null_projector(o))


def kappa_value(g: np.ndarray, o: np.ndarray) -> float:
    return op_norm(g @ np.linalg.pinv(o, rcond=1e-10))


def augmented(o: np.ndarray, rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return o
    return np.vstack([o] + rows)


def certificate(g: np.ndarray, o: np.ndarray, rows: list[np.ndarray]) -> dict:
    oc = augmented(o, rows)
    beta = beta_value(g, oc)
    kappa = kappa_value(g, oc)
    scale = op_norm(oc)
    return {
        "beta": beta,
        "kappa": kappa,
        "observation_norm": scale,
        "j_cert": beta + CERT_LAMBDA * kappa * scale,
    }


def actual_observation(theta: np.ndarray, selected: list[int]) -> np.ndarray:
    rows = [(V0 @ theta) * V0]
    for idx in selected:
        row = CANDIDATE_ROWS[idx]
        rows.append((row @ theta) * row)
    return np.asarray(rows)


def train(selected: list[int], steps: int, learning_rate: float = 0.05) -> np.ndarray:
    theta = np.zeros(Q)
    deltas = SOURCE_ETAS - ETA_BAR
    base_var = float(np.mean((deltas @ V0) ** 2))
    candidate_vars = [
        float(np.mean((deltas @ row) ** 2)) for row in CANDIDATE_ROWS
    ]
    for _ in range(steps):
        grad = np.mean([risk_grad(theta, eta) for eta in SOURCE_ETAS], axis=0)
        base_value = V0 @ theta
        grad += 2.0 * BASE_LAMBDA * base_var * base_value * V0
        for idx in selected:
            row = CANDIDATE_ROWS[idx]
            value = row @ theta
            grad += 2.0 * CANDIDATE_LAMBDA * candidate_vars[idx] * value * row
        theta -= learning_rate * grad
    return theta


def target_optimum() -> tuple[np.ndarray, float]:
    theta = np.zeros(Q)
    target_eta = np.zeros(Q)
    for _ in range(16000):
        theta -= 0.05 * risk_grad(theta, target_eta)
    return theta, risk(theta, target_eta)


TARGET_THETA, TARGET_OPT_RISK = target_optimum()


def evaluate(theta: np.ndarray, selected: list[int], g_true: np.ndarray) -> dict:
    actual_o = actual_observation(theta, selected)
    actual_cert = certificate(g_true, actual_o[:1], list(actual_o[1:]))
    unit_o = V0.reshape(1, -1)
    unit_rows = [CANDIDATE_ROWS[idx].reshape(1, -1) for idx in selected]
    selection_cert = certificate(g_true, unit_o, unit_rows)
    target_eta = np.zeros(Q)
    return {
        "theta": theta.tolist(),
        "selected": selected,
        "selected_names": [CANDIDATE_NAMES[idx] for idx in selected],
        "source_risk": float(np.mean([risk(theta, eta) for eta in SOURCE_ETAS])),
        "target_risk": risk(theta, target_eta),
        "target_excess": risk(theta, target_eta) - TARGET_OPT_RISK,
        "augmented_observation_rank": int(np.linalg.matrix_rank(actual_o)),
        "remaining_blind_dimension": int(Q - np.linalg.matrix_rank(actual_o)),
        "actual_row_scales": [float(V0 @ theta)]
        + [float(CANDIDATE_ROWS[idx] @ theta) for idx in selected],
        **actual_cert,
        "selection_beta": selection_cert["beta"],
        "selection_kappa": selection_cert["kappa"],
        "selection_j_cert": selection_cert["j_cert"],
    }


def select_tsr(g_hat: np.ndarray, o0: np.ndarray) -> int:
    scores = []
    for idx, row in enumerate(CANDIDATE_ROWS):
        score = certificate(g_hat, o0, [row.reshape(1, -1)])["j_cert"]
        scores.append((score, idx))
    return min(scores)[1]


def select_raw_magnitude(g_hat: np.ndarray) -> int:
    magnitudes = [
        (float(np.linalg.norm(g_hat @ row)), idx)
        for idx, row in enumerate(CANDIDATE_ROWS)
    ]
    return max(magnitudes)[1]


def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    theta_probe = train([], TOTAL_STATISTIC_GRAD_BUDGET)
    g_fd = finite_response(theta_probe, ETA_BAR, 1e-3)
    g_half = finite_response(theta_probe, ETA_BAR, 5e-4)
    response_noise = 0.01 * rng.normal(size=g_fd.shape)
    g_hat = g_fd + response_noise
    o0 = V0.reshape(1, -1)
    tsr_idx = select_tsr(g_hat, o0)
    magnitude_idx = select_raw_magnitude(g_hat)
    random_idx = int(rng.integers(len(CANDIDATE_ROWS)))
    selection_scores = [
        certificate(g_hat, o0, [row.reshape(1, -1)])["j_cert"]
        for row in CANDIDATE_ROWS
    ]
    methods = {
        "baseline": [],
        "random": [random_idx],
        "raw_magnitude": [magnitude_idx],
        "tsr": [tsr_idx],
        "full_alignment": list(range(len(CANDIDATE_ROWS))),
    }
    rows = []
    for name, selected in methods.items():
        cost = 1 + len(selected)
        steps = max(100, TOTAL_STATISTIC_GRAD_BUDGET // cost)
        theta = train(selected, steps)
        item = evaluate(theta, selected, finite_response(theta, ETA_BAR, 1e-3))
        item.update({
            "seed": seed,
            "method": name,
            "steps": steps,
            "statistic_gradient_budget": steps * cost,
        })
        rows.append(item)
    raw_norms = [float(np.linalg.norm(g_hat @ row)) for row in CANDIDATE_ROWS]
    return {
        "seed": seed,
        "rows": rows,
        "tsr_selection": CANDIDATE_NAMES[tsr_idx],
        "raw_magnitude_selection": CANDIDATE_NAMES[magnitude_idx],
        "raw_response_norms": raw_norms,
        "selection_j_cert": selection_scores,
        "finite_difference_sensitivity": op_norm(g_fd - g_half),
        "source_response_noise": op_norm(response_noise),
        "off_diagonal_ratio": float(
            np.linalg.norm(B - np.diag(np.diag(B))) / np.linalg.norm(B)
        ),
    }


def summarize(records: list[dict]) -> list[dict]:
    flat = [row for rec in records for row in rec["rows"]]
    out = []
    for method in ["baseline", "random", "raw_magnitude", "tsr", "full_alignment"]:
        group = [row for row in flat if row["method"] == method]
        item = {"method": method}
        for key in ["beta", "kappa", "j_cert", "target_excess",
                    "source_risk", "remaining_blind_dimension"]:
            values = np.asarray([row[key] for row in group], dtype=float)
            item[key] = float(values.mean())
            item[key + "_std"] = float(values.std())
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    records = [run_seed(seed) for seed in args.seeds]
    output = {
        "configuration": {
            "q": Q,
            "rank_O0": 1,
            "candidate_names": CANDIDATE_NAMES,
            "trap_epsilon": TRAP_EPS,
            "cert_lambda": CERT_LAMBDA,
            "target_opt_risk": TARGET_OPT_RISK,
            "target_opt_theta": TARGET_THETA.tolist(),
            "total_statistic_gradient_budget": TOTAL_STATISTIC_GRAD_BUDGET,
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
