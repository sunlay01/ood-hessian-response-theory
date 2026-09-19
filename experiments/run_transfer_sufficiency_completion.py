"""Minimal falsification experiments for the transfer-sufficiency completion loop.

This script is a numerical companion, not a proof.  It has three deliberately
small cases:

1. an exactly known linear operator, where the unrestricted completion can be
   checked against sigma_(r+1);
2. the two-dimensional IRMv1 witness from theory/13;
3. an empirical logistic-loss model, where G and the IRM statistic derivative
   are obtained by finite differences of source distributions.

The script reports source-span estimates, true beta/kappa, finite-difference
and split-sample sensitivity, and target-risk or local-bound consequences.  It
never treats a source response matrix as the target operator without naming the
coverage/estimation assumptions.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def op_norm(a: np.ndarray) -> float:
    """Euclidean induced operator norm, with the zero-map convention."""
    a = np.asarray(a, dtype=float)
    if a.size == 0 or min(a.shape) == 0:
        return 0.0
    return float(np.linalg.svd(a, compute_uv=False)[0])


def null_projector(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto ker(a), robust for empty row blocks."""
    a = np.asarray(a, dtype=float)
    q = a.shape[1]
    if a.shape[0] == 0:
        return np.eye(q)
    p = np.linalg.pinv(a, rcond=tol) @ a
    p = (p + p.T) / 2.0
    return (np.eye(q) - p + (np.eye(q) - p).T) / 2.0


def beta_value(g: np.ndarray, o: np.ndarray) -> float:
    return op_norm(g @ null_projector(o))


def kappa_value(g: np.ndarray, o: np.ndarray) -> float:
    if o.shape[0] == 0:
        return float("inf")
    return op_norm(g @ np.linalg.pinv(o))


def coverage_error(q_frame: np.ndarray, o: np.ndarray) -> float:
    """||(I-P_S)P_0|| from theory/15."""
    p_s = q_frame @ np.linalg.pinv(q_frame)
    p_s = (p_s + p_s.T) / 2.0
    return op_norm((np.eye(q_frame.shape[0]) - p_s) @ null_projector(o))


def augmented(o: np.ndarray, c: np.ndarray | None) -> np.ndarray:
    if c is None or c.size == 0:
        return np.asarray(o, dtype=float)
    return np.vstack([o, c])


def select_dictionary(
    g_est: np.ndarray,
    o0: np.ndarray,
    bank: list[np.ndarray],
    rank_budget: int,
    kappa_weight: float = 0.0,
) -> tuple[list[int], np.ndarray, float]:
    """Exact restricted completion for a small declared statistic dictionary."""
    best = ([], np.zeros((0, o0.shape[1])), beta_value(g_est, o0))
    best_score = best[2] + kappa_weight * kappa_value(g_est, o0)
    for k in range(1, min(rank_budget, len(bank)) + 1):
        for idx in itertools.combinations(range(len(bank)), k):
            c = np.vstack([bank[i] for i in idx])
            value = beta_value(g_est, augmented(o0, c))
            score = value + kappa_weight * kappa_value(g_est, augmented(o0, c))
            if score < best_score - 1e-12:
                best = (list(idx), c, value)
                best_score = score
    return best


def unrestricted_completion(g: np.ndarray, o0: np.ndarray, rank_budget: int):
    """Top right-singular rows of G restricted to ker(O0)."""
    p0 = null_projector(o0)
    e = g @ p0
    _, s, vh = np.linalg.svd(e, full_matrices=True)
    q = o0.shape[1]
    s_pad = np.pad(s, (0, max(0, q - len(s))))
    r = min(rank_budget, q)
    c = vh[:r]
    return c, beta_value(g, augmented(o0, c)), float(s_pad[r])


def linear_case(seed: int, rank_budget: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    q, z = 6, 5
    o0 = rng.normal(size=(2, q))
    g = rng.normal(size=(z, q))
    # Q=I is exact source tangent coverage in this controlled case.
    q_frame = np.eye(q)
    noise = 0.015 * rng.normal(size=g.shape)
    g_hat = g + noise
    bank = [np.eye(q)[j : j + 1] for j in range(q)]
    picked, c_tsr, _ = select_dictionary(g_hat, o0, bank, rank_budget)
    c_oracle, beta_oracle, sigma_next = unrestricted_completion(
        g, o0, rank_budget
    )
    random_betas = []
    for _ in range(24):
        rows = []
        for _ in range(rank_budget):
            row = rng.normal(size=(1, q))
            row /= max(np.linalg.norm(row), 1e-15)
            rows.append(row)
        random_betas.append(beta_value(g, augmented(o0, np.vstack(rows))))
    beta_before = beta_value(g, o0)
    beta_tsr = beta_value(g, augmented(o0, c_tsr))
    beta_hat_before = beta_value(g_hat, o0)
    beta_hat_tsr = beta_value(g_hat, augmented(o0, c_tsr))
    return {
        "seed": seed,
        "beta_before": beta_before,
        "beta_tsr": beta_tsr,
        "beta_hat_before": beta_hat_before,
        "beta_hat_tsr": beta_hat_tsr,
        "beta_hat_upper_bound": beta_hat_before + op_norm(noise),
        "beta_oracle": beta_oracle,
        "sigma_r_plus_1": sigma_next,
        "oracle_abs_error": abs(beta_oracle - sigma_next),
        "beta_random_mean": float(np.mean(random_betas)),
        "beta_random_std": float(np.std(random_betas)),
        "kappa_before": kappa_value(g, o0),
        "kappa_tsr": kappa_value(g, augmented(o0, c_tsr)),
        "kappa_hat_before": kappa_value(g_hat, o0),
        "kappa_hat_tsr": kappa_value(g_hat, augmented(o0, c_tsr)),
        "coverage_exact": coverage_error(q_frame, o0),
        "source_response_noise": op_norm(noise),
        "selected_coordinates": picked,
        "bound_transfer_before_eps_02": 0.2 * beta_before,
        "bound_transfer_tsr_eps_02": 0.2 * beta_tsr,
    }


def irm_case(eps: float = 0.1) -> dict:
    # This is exactly the O_IRM and G_IRM displayed in theory/13.
    o0 = np.ones((2, 2))
    g = np.eye(2)
    h = np.array([1.0, -1.0])
    unit_sum = np.array([[1.0, 1.0]]) / np.sqrt(2.0)
    unit_diff = np.array([[1.0, -1.0]]) / np.sqrt(2.0)
    bank = [unit_sum, unit_diff]
    picked, c_tsr, _ = select_dictionary(g, o0, bank, rank_budget=1)
    beta_before = beta_value(g, o0)
    beta_after = beta_value(g, augmented(o0, c_tsr))
    random_betas = [
        beta_value(g, augmented(o0, bank[i])) for i in (0, 1)
    ]
    # The target-risk gap is for the fixed mixed zero-residual point.  The
    # transfer-bound term is the consequence certified by completion; the
    # fixed-point target gap is intentionally not claimed to change by itself.
    source_excess = 1.0
    target_excess_fixed = (1.0 + eps) ** 2
    gap_fixed = target_excess_fixed - source_excess
    theta_projected = np.zeros(2)  # hard C theta=0 counterfactual branch
    target_excess_projected = 0.5 * np.linalg.norm(theta_projected + eps * h) ** 2
    return {
        "beta_before": beta_before,
        "beta_tsr": beta_after,
        "beta_random_mean": float(np.mean(random_betas)),
        "kappa_before": kappa_value(g, o0),
        "kappa_tsr": kappa_value(g, augmented(o0, c_tsr)),
        "selected_statistic": "sum" if picked == [0] else "difference",
        "bound_transfer_before_eps_01": eps * beta_before * np.linalg.norm(h),
        "bound_transfer_tsr_eps_01": eps * beta_after * np.linalg.norm(h),
        "fixed_theta_target_source_gap": gap_fixed,
        "projected_branch_target_excess": float(target_excess_projected),
        "warning": "fixed-theta target risk is unchanged by an operator certificate; projected branch is counterfactual",
    }


def logistic_data(eta: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Common-random-number logistic environment family."""
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([-1.0, 1.0]), size=n)
    x0 = 1.5 * y + rng.normal(size=n)
    x1 = (0.35 + eta[0]) * y + rng.normal(size=n)
    x2 = np.exp(eta[1]) * rng.normal(size=n)
    return np.column_stack([x0, x1, x2]), y


def logistic_stats(w: np.ndarray, eta: np.ndarray, n: int, seed: int):
    x, y = logistic_data(eta, n, seed)
    t = y * (x @ w)
    s = 1.0 / (1.0 + np.exp(np.clip(t, -60.0, 60.0)))
    loss = float(np.logaddexp(0.0, -t).mean())
    grad = np.mean((-y * s)[:, None] * x, axis=0)
    weights = s * (1.0 - s)
    hess = (x.T * weights) @ x / n
    return loss, grad, hess


def irm_q(w: np.ndarray, eta: np.ndarray, n: int, seed: int) -> float:
    x, y = logistic_data(eta, n, seed)
    t = y * (x @ w)
    s = 1.0 / (1.0 + np.exp(np.clip(t, -60.0, 60.0)))
    return float(np.mean(-s * t))


def transfer_vector(w: np.ndarray, eta: np.ndarray, n: int, seed: int) -> np.ndarray:
    _, grad, hess = logistic_stats(w, eta, n, seed)
    return np.concatenate([grad, 0.5 * hess.reshape(-1)])


def finite_response(w, eta0, delta, n, seed):
    q = len(eta0)
    cols = []
    for j in range(q):
        e = np.zeros(q)
        e[j] = 1.0
        cols.append(
            (transfer_vector(w, eta0 + delta * e, n, seed)
             - transfer_vector(w, eta0 - delta * e, n, seed))
            / (2.0 * delta)
        )
    return np.column_stack(cols)


def logistic_case(seed: int, n: int = 1800) -> dict:
    eta0 = np.zeros(2)
    source_etas = [np.array([0.45, 0.0]), np.array([-0.45, 0.0])]

    def objective(w):
        vals = [logistic_stats(w, e, n, seed + 10 + j) for j, e in enumerate(source_etas)]
        return float(np.mean([v[0] for v in vals])), np.mean([v[1] for v in vals], axis=0)

    opt = minimize(lambda w: objective(w), np.zeros(3), jac=True, method="BFGS")
    w = opt.x
    delta = 2e-3
    g_est = finite_response(w, eta0, delta, n, seed + 77)

    # Mean IRMv1 residual derivative: a scalar source statistic, deliberately
    # leaving a one-dimensional local kernel in the two-dimensional eta space.
    rows = []
    for j, eta in enumerate(source_etas):
        row = []
        for i in range(2):
            e = np.zeros(2)
            e[i] = 1.0
            row.append(
                (irm_q(w, eta + delta * e, n, seed + 10 + j)
                 - irm_q(w, eta - delta * e, n, seed + 10 + j))
                / (2.0 * delta)
            )
        rows.append(row)
    o0 = np.mean(rows, axis=0, keepdims=True)
    o0 /= max(np.linalg.norm(o0), 1e-15)

    # A declared source-statistic bank: corr moment, nuisance-scale moment,
    # their mixed moment, and a redundant copy of the current IRM row.
    bank = [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 1.0]]) / np.sqrt(2.0),
        o0.copy(),
    ]
    # The logistic IRM row is almost collinear with the correlation moment.  A
    # kappa-aware score avoids selecting that numerically unstable completion.
    picked, c_tsr, _ = select_dictionary(
        g_est, o0, bank, rank_budget=1, kappa_weight=0.1
    )
    beta_before = beta_value(g_est, o0)
    beta_tsr = beta_value(g_est, augmented(o0, c_tsr))
    random_betas = [
        beta_value(g_est, augmented(o0, bank[i])) for i in range(len(bank))
    ]

    # Finite-difference sensitivity is a declared linearization error proxy;
    # split-sample sensitivity is a declared statistical error proxy.
    g_half = finite_response(w, eta0, delta, max(500, n // 2), seed + 177)
    g_delta2 = finite_response(w, eta0, 2.0 * delta, n, seed + 77)
    eps_stat = op_norm(g_est - g_half)
    eps_fd = op_norm(g_est - g_delta2)

    # Worst-case local direction in the current kernel and an actual target
    # logistic excess-risk check for that direction.
    _, _, vh = np.linalg.svd(o0, full_matrices=True)
    h = vh[-1]
    h /= np.linalg.norm(h)
    target_eta = 0.20 * h
    target_seed = seed + 900

    def target_objective(v):
        loss, grad, _ = logistic_stats(v, target_eta, n, target_seed)
        return loss, grad

    target_opt = minimize(lambda v: target_objective(v), np.zeros(3), jac=True, method="BFGS")
    target_loss_at_w = target_objective(w)[0]
    target_excess = target_loss_at_w - target_opt.fun
    source_loss = objective(w)[0]
    return {
        "seed": seed,
        "optimizer_success": bool(opt.success and target_opt.success),
        "beta_before": beta_before,
        "beta_tsr": beta_tsr,
        "beta_random_mean": float(np.mean(random_betas)),
        "kappa_before": kappa_value(g_est, o0),
        "kappa_tsr": kappa_value(g_est, augmented(o0, c_tsr)),
        "selected_statistic": ["corr", "nuisance", "mixed", "redundant_irm"][picked[0]],
        "coverage_exact": 0.0,  # Q=I_2 in the finite-difference tangent model
        "epsilon_stat_proxy": eps_stat,
        "epsilon_fd_proxy": eps_fd,
        "source_loss": source_loss,
        "target_excess_at_irm_solution": float(target_excess),
        "local_bound_transfer_eps_02": 0.2 * beta_before,
        "local_bound_transfer_tsr_eps_02": 0.2 * beta_tsr,
        "irm_row": o0.reshape(-1).tolist(),
    }


def summarize(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run(seeds: list[int]) -> dict:
    linear = [linear_case(s) for s in seeds]
    irm = irm_case()
    logistic = [logistic_case(s) for s in seeds]
    numeric_keys = [
        "beta_before", "beta_tsr", "beta_oracle", "sigma_r_plus_1",
        "beta_hat_before", "beta_hat_tsr", "beta_hat_upper_bound",
        "oracle_abs_error", "beta_random_mean", "kappa_before", "kappa_tsr",
        "kappa_hat_before", "kappa_hat_tsr",
        "coverage_exact", "source_response_noise",
    ]
    log_keys = [
        "beta_before", "beta_tsr", "beta_random_mean", "kappa_before",
        "kappa_tsr", "coverage_exact", "epsilon_stat_proxy", "epsilon_fd_proxy",
        "target_excess_at_irm_solution", "local_bound_transfer_eps_02",
        "local_bound_transfer_tsr_eps_02",
    ]
    return {
        "seeds": seeds,
        "linear": linear,
        "linear_summary": summarize(linear, numeric_keys),
        "irm": irm,
        "logistic": logistic,
        "logistic_summary": summarize(logistic, log_keys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.seeds)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
