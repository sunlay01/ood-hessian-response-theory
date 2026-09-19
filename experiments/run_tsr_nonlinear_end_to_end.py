"""High-dimensional nonlinear end-to-end probe for source-side TSR.

The model is a small nonlinear population risk, not a claim about a benchmark:

    R(theta, eta) = 1/2 ||theta - (a_inv + B eta)||^2
                    + alpha(eta) * sum_j log cosh(theta_j).

There are q=6 environment tangent coordinates.  The baseline statistic has
rank one, the candidate bank has four spurious-coordinate statistics, and the
rank budget is swept over r in {0, 1, 2}.  The source tangent frame is exact
in this controlled experiment, while G is still estimated by finite
differences plus small source-response noise.

Every method is jointly retrained from the same initialization.  The step
budget is matched in statistic-gradient evaluations: a method using k extra
rows receives floor(B/(1+k)) updates.  This is a mechanism probe, not a
generalization benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


Q = 6
AINV = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
B = np.diag([0.20, 3.00, 1.20, 0.80, 0.60, 0.25])
ETA_BAR = np.array([0.0, 0.8, 0.8, 0.8, 0.8, 0.5])
SOURCE_DELTA = 0.35
TOTAL_STATISTIC_GRAD_BUDGET = 6000
BASE_LAMBDA = 20.0
CANDIDATE_LAMBDA = 30.0
CERT_LAMBDA = 0.05


def source_environments() -> np.ndarray:
    envs = [ETA_BAR.copy()]
    for j in range(Q):
        e = np.zeros(Q)
        e[j] = SOURCE_DELTA
        envs.extend([ETA_BAR + e, ETA_BAR - e])
    return np.asarray(envs)


SOURCE_ETAS = source_environments()
TANGENT_FRAME = np.eye(Q)


def log_cosh(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, -x) - np.log(2.0)


def alpha_eta(eta: np.ndarray) -> float:
    return 0.25 + 0.05 * np.tanh(eta[0])


def alpha_prime_eta(eta: np.ndarray) -> np.ndarray:
    out = np.zeros(Q)
    out[0] = 0.05 * (1.0 - np.tanh(eta[0]) ** 2)
    return out


def target_center(eta: np.ndarray) -> np.ndarray:
    return AINV + B @ eta


def risk(theta: np.ndarray, eta: np.ndarray) -> float:
    residual = theta - target_center(eta)
    return float(0.5 * np.dot(residual, residual)
                 + alpha_eta(eta) * np.sum(log_cosh(theta)))


def risk_grad(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    residual = theta - target_center(eta)
    return residual + alpha_eta(eta) * np.tanh(theta)


def risk_hessian(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    diagonal = 1.0 + alpha_eta(eta) * (1.0 - np.tanh(theta) ** 2)
    return np.diag(diagonal)


def transfer_vector(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.concatenate([risk_grad(theta, eta),
                           0.5 * risk_hessian(theta, eta).reshape(-1)])


def finite_response(theta: np.ndarray, eta0: np.ndarray, step: float) -> np.ndarray:
    cols = []
    for j in range(Q):
        e = np.zeros(Q)
        e[j] = 1.0
        cols.append((transfer_vector(theta, eta0 + step * e)
                     - transfer_vector(theta, eta0 - step * e)) / (2.0 * step))
    return np.column_stack(cols)


def op_norm(a: np.ndarray) -> float:
    if a.size == 0 or min(a.shape) == 0:
        return 0.0
    return float(np.linalg.svd(a, compute_uv=False)[0])


def null_projector(a: np.ndarray) -> np.ndarray:
    q = a.shape[1]
    if a.shape[0] == 0:
        return np.eye(q)
    p = np.linalg.pinv(a, rcond=1e-10) @ a
    p = (p + p.T) / 2.0
    return (np.eye(q) - p + (np.eye(q) - p).T) / 2.0


def beta_value(g: np.ndarray, o: np.ndarray) -> float:
    return op_norm(g @ null_projector(o))


def kappa_value(g: np.ndarray, o: np.ndarray) -> float:
    return op_norm(g @ np.linalg.pinv(o, rcond=1e-10))


def normalized_baseline_row(theta: np.ndarray) -> np.ndarray:
    row = np.zeros((1, Q))
    row[0, 5] = 1.0 if abs(theta[5]) < 1e-10 else theta[5]
    row /= np.linalg.norm(row)
    return row


def candidate_bank() -> list[np.ndarray]:
    return [np.eye(Q)[j : j + 1] for j in (1, 2, 3, 4)]


def augmented(o: np.ndarray, c: np.ndarray | None) -> np.ndarray:
    if c is None or c.size == 0:
        return o
    return np.vstack([o, c])


def certificate(g: np.ndarray, o: np.ndarray, c: np.ndarray | None) -> dict:
    oc = augmented(o, c)
    beta = beta_value(g, oc)
    kappa = kappa_value(g, oc)
    scale = op_norm(oc)
    return {
        "beta": beta,
        "kappa": kappa,
        "observation_norm": scale,
        "j_cert": beta + CERT_LAMBDA * kappa * scale,
    }


def certificate_scale_error(
    g: np.ndarray, o: np.ndarray, c: np.ndarray | None
) -> float:
    base = certificate(g, o, c)["j_cert"]
    scaled = certificate(
        g,
        2.0 * o,
        None if c is None else 2.0 * c,
    )["j_cert"]
    return abs(base - scaled)


def select_rows(g_hat: np.ndarray, o0: np.ndarray, rank_budget: int) -> list[int]:
    bank = candidate_bank()
    best: tuple[list[int], float] = ([], certificate(g_hat, o0, None)["j_cert"])
    for k in range(1, min(rank_budget, len(bank)) + 1):
        for idx in itertools.combinations(range(len(bank)), k):
            c = np.vstack([bank[i] for i in idx])
            score = certificate(g_hat, o0, c)["j_cert"]
            if score < best[1] - 1e-12:
                best = (list(idx), score)
    return best[0]


def statistic_penalty_grad(theta: np.ndarray, selected: list[int]) -> np.ndarray:
    """Gradient of centered source-statistic penalties."""
    deltas = SOURCE_ETAS - ETA_BAR
    grad = np.zeros(Q)
    # Baseline rank-one statistic: S0(theta,eta)=theta_5*(eta_5-eta_bar_5).
    variance_5 = float(np.mean(deltas[:, 5] ** 2))
    grad[5] += 2.0 * BASE_LAMBDA * variance_5 * theta[5]
    # Candidate j uses Sj(theta,eta)=theta_j*(eta_j-eta_bar_j).
    for bank_index in selected:
        j = (1, 2, 3, 4)[bank_index]
        variance_j = float(np.mean(deltas[:, j] ** 2))
        grad[j] += 2.0 * CANDIDATE_LAMBDA * variance_j * theta[j]
    return grad


def train(selected: list[int], steps: int, learning_rate: float = 0.05) -> np.ndarray:
    theta = np.zeros(Q)
    for _ in range(steps):
        grad = np.mean([risk_grad(theta, eta) for eta in SOURCE_ETAS], axis=0)
        grad += statistic_penalty_grad(theta, selected)
        theta -= learning_rate * grad
    return theta


def target_optimum() -> tuple[np.ndarray, float]:
    theta = np.zeros(Q)
    for _ in range(12000):
        theta -= 0.05 * risk_grad(theta, np.zeros(Q))
    return theta, risk(theta, np.zeros(Q))


TARGET_THETA, TARGET_OPT_RISK = target_optimum()


def evaluate(theta: np.ndarray, selected: list[int], g_true: np.ndarray) -> dict:
    o0 = normalized_baseline_row(theta)
    bank = candidate_bank()
    c = np.vstack([bank[i] for i in selected]) if selected else None
    o_aug = augmented(o0, c)
    cert = certificate(g_true, o0, c)
    target_eta = np.zeros(Q)
    target_risk = risk(theta, target_eta)
    source_risk = float(np.mean([risk(theta, eta) for eta in SOURCE_ETAS]))
    return {
        "theta": theta.tolist(),
        "source_risk": source_risk,
        "target_risk": target_risk,
        "target_excess": target_risk - TARGET_OPT_RISK,
        "augmented_observation_rank": int(np.linalg.matrix_rank(o_aug)),
        "remaining_blind_dimension": int(Q - np.linalg.matrix_rank(o_aug)),
        "selected": selected,
        **cert,
    }


def run_seed(seed: int, rank_budgets: list[int]) -> dict:
    rng = np.random.default_rng(seed)
    theta_probe = train([], TOTAL_STATISTIC_GRAD_BUDGET)
    eta0 = ETA_BAR
    g_fd = finite_response(theta_probe, eta0, 1e-3)
    g_fd_half = finite_response(theta_probe, eta0, 5e-4)
    response_noise = 0.01 * rng.normal(size=g_fd.shape)
    g_hat = g_fd + response_noise
    o0 = normalized_baseline_row(theta_probe)
    assert np.linalg.matrix_rank(o0) == 1
    bank = candidate_bank()
    baseline_cert = certificate(g_fd, o0, None)
    rows = []
    for r in rank_budgets:
        tsr_selected = select_rows(g_hat, o0, r)
        random_selected = sorted(rng.choice(len(bank), size=r, replace=False).tolist())
        methods = {
            "baseline": [],
            "random": random_selected,
            "tsr": tsr_selected,
            "full_alignment": list(range(len(bank))),
        }
        for name, selected in methods.items():
            # One baseline statistic-gradient evaluation plus one per active
            # candidate row defines the matched budget.
            row_cost = 1 + len(selected)
            steps = max(100, TOTAL_STATISTIC_GRAD_BUDGET // row_cost)
            theta = train(selected, steps)
            item = evaluate(theta, selected, g_fd)
            item.update({
                "seed": seed,
                "rank_budget": r,
                "method": name,
                "steps": steps,
                "statistic_gradient_budget": steps * row_cost,
            })
            rows.append(item)
    return {
        "seed": seed,
        "rank_budgets": rank_budgets,
        "baseline_certificate_at_probe": baseline_cert,
        "certificate_scale_error": certificate_scale_error(g_fd, o0, None),
        "finite_difference_sensitivity": op_norm(g_fd - g_fd_half),
        "source_response_noise": op_norm(response_noise),
        "rows": rows,
    }


def summarize(records: list[dict]) -> list[dict]:
    flat = [row for rec in records for row in rec["rows"]]
    keys = ["beta", "kappa", "j_cert", "source_risk", "target_risk",
            "target_excess",
            "observation_norm", "steps", "statistic_gradient_budget"]
    out = []
    for (r, method), group in itertools.groupby(
        sorted(flat, key=lambda x: (x["rank_budget"], x["method"])),
        key=lambda x: (x["rank_budget"], x["method"]),
    ):
        group = list(group)
        item = {"rank_budget": r, "method": method}
        for key in keys:
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
    rank_budgets = [0, 1, 2]
    records = [run_seed(seed, rank_budgets) for seed in args.seeds]
    output = {
        "configuration": {
            "q": Q,
            "rank_O0": 1,
            "rank_budgets": rank_budgets,
            "candidate_bank": [1, 2, 3, 4],
            "total_statistic_gradient_budget": TOTAL_STATISTIC_GRAD_BUDGET,
            "cert_lambda": CERT_LAMBDA,
            "base_lambda": BASE_LAMBDA,
            "candidate_lambda": CANDIDATE_LAMBDA,
            "target_opt_risk": TARGET_OPT_RISK,
            "target_opt_theta": TARGET_THETA.tolist(),
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
