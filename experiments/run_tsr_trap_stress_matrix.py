"""Finite-coverage stress matrix for the TSR selection-trap probe.

This is a deliberately narrow stress test of the preceding population
construction.  It does not claim to be a finite-sample neural-network
experiment.  Instead it replaces the exact source response by

    G_hat = G P_alpha + noise / sqrt(n),

where P_alpha is a rank-(q-1) source-coverage projector.  The missing direction
is cos(alpha) v_2 + sin(alpha) v_3, so alpha=0 hides the dominant blind
direction and alpha=90 hides the weaker v_3 direction.  The matrix sweeps
sample size, coverage angle, and response noise, then measures TSR row
selection and the target excess induced by the selected row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from . import run_tsr_nondiagonal_selection_trap as trap
except ImportError:  # direct execution: python experiments/run_tsr_trap_stress_matrix.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_tsr_nondiagonal_selection_trap as trap


DEFAULT_SAMPLE_SIZES = [8, 32, 128, 512]
DEFAULT_COVERAGE_ANGLES = [0.0, 30.0, 45.0, 60.0, 75.0, 90.0]
DEFAULT_NOISE_LEVELS = [0.2, 0.5, 1.0]
DEFAULT_REPLICATES = 64


def coverage_projector(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    missing = np.cos(angle) * trap.V2 + np.sin(angle) * trap.V3
    return np.eye(trap.Q) - np.outer(missing, missing)


def estimate_response(
    g_true: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    coverage_angle: float,
    noise_level: float,
) -> tuple[np.ndarray, float, float]:
    projector = coverage_projector(coverage_angle)
    coverage_defect = trap.op_norm(g_true @ (np.eye(trap.Q) - projector))
    noise_scale = noise_level * trap.op_norm(g_true) / np.sqrt(sample_size)
    estimate = g_true @ projector + noise_scale * rng.normal(size=g_true.shape)
    return estimate, coverage_defect, noise_scale


def run_matrix(
    sample_sizes: list[int],
    coverage_angles: list[float],
    noise_levels: list[float],
    replicates: int,
    seed: int,
) -> dict:
    theta_probe = trap.train([], trap.TOTAL_STATISTIC_GRAD_BUDGET)
    g_true = trap.finite_response(theta_probe, trap.ETA_BAR, 1e-3)
    o0 = trap.V0.reshape(1, -1)
    ideal_scores = [
        trap.certificate(g_true, o0, [row.reshape(1, -1)])["j_cert"]
        for row in trap.CANDIDATE_ROWS
    ]
    sorted_scores = np.sort(np.asarray(ideal_scores))
    ideal_score_gap = float(sorted_scores[1] - sorted_scores[0])

    target_excess_by_candidate = {}
    for idx in range(len(trap.CANDIDATE_ROWS)):
        steps = trap.TOTAL_STATISTIC_GRAD_BUDGET // 2
        theta = trap.train([idx], steps)
        target_excess_by_candidate[trap.CANDIDATE_NAMES[idx]] = trap.evaluate(
            theta,
            [idx],
            trap.finite_response(theta, trap.ETA_BAR, 1e-3),
        )["target_excess"]

    rng = np.random.default_rng(seed)
    cells = []
    for sample_size in sample_sizes:
        for coverage_angle in coverage_angles:
            for noise_level in noise_levels:
                tsr_hits = 0
                raw_hits = 0
                tsr_target_excess = []
                raw_target_excess = []
                coverage_defect = 0.0
                noise_scale = 0.0
                for _ in range(replicates):
                    estimate, coverage_defect, noise_scale = estimate_response(
                        g_true,
                        rng,
                        sample_size,
                        coverage_angle,
                        noise_level,
                    )
                    tsr_idx = trap.select_tsr(estimate, o0)
                    raw_idx = trap.select_raw_magnitude(estimate)
                    tsr_hits += int(tsr_idx == 1)
                    raw_hits += int(raw_idx == 0)
                    tsr_target_excess.append(
                        target_excess_by_candidate[trap.CANDIDATE_NAMES[tsr_idx]]
                    )
                    raw_target_excess.append(
                        target_excess_by_candidate[trap.CANDIDATE_NAMES[raw_idx]]
                    )
                total_proxy_error = coverage_defect + noise_scale
                cells.append({
                    "sample_size": sample_size,
                    "coverage_angle_deg": coverage_angle,
                    "response_noise_level": noise_level,
                    "replicates": replicates,
                    "tsr_selection_accuracy": tsr_hits / replicates,
                    "raw_magnitude_trap_accuracy": raw_hits / replicates,
                    "mean_tsr_target_excess": float(np.mean(tsr_target_excess)),
                    "std_tsr_target_excess": float(np.std(tsr_target_excess)),
                    "mean_raw_target_excess": float(np.mean(raw_target_excess)),
                    "std_raw_target_excess": float(np.std(raw_target_excess)),
                    "coverage_defect_norm": coverage_defect,
                    "sampling_noise_norm": noise_scale,
                    "total_proxy_error": total_proxy_error,
                    "error_to_score_gap": total_proxy_error / ideal_score_gap,
                })

    return {
        "configuration": {
            "sample_sizes": sample_sizes,
            "coverage_angles_deg": coverage_angles,
            "response_noise_levels": noise_levels,
            "replicates": replicates,
            "seed": seed,
            "coverage_definition": (
                "missing direction cos(alpha) v2 + sin(alpha) v3; "
                "P_alpha projects onto its orthogonal complement"
            ),
            "ideal_selection_scores": ideal_scores,
            "ideal_score_gap": ideal_score_gap,
            "target_excess_by_candidate": target_excess_by_candidate,
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument(
        "--coverage-angles",
        type=float,
        nargs="+",
        default=DEFAULT_COVERAGE_ANGLES,
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    output = run_matrix(
        args.sample_sizes,
        args.coverage_angles,
        args.noise_levels,
        args.replicates,
        args.seed,
    )
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Running from the repository root puts this file's directory on sys.path,
    # but retaining the explicit path makes direct module execution robust.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
