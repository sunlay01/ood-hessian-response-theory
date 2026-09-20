"""Source extrapolation training and finite-step guard (v3), with v2 controls.

This is an empirical source-contrast experiment, NOT a target-risk certifier.
New robust methods do not optimize or guard IRM. Legacy methods remain controls.
Response estimation uses fixed source-only calibration examples.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from run_cmnist_signed_repair import (MLP, REFERENCE, build_official_envs,
    evaluate, irm_penalty, mean_nll)
from response_certificate_core import source_terms, response, soft_gate, contrasts

ALGORITHM_VERSION = "3.0-source-extrapolation-response"
ROBUST_METHODS = ("robust", "robust_response", "robust_guard")


def risk_envelope(risks, coverage, smoothing):
    """Support function of an L2 ball of affine environment coefficients.

    Smoothing is upward (not subtracted), preserving the envelope inequality.
    Coverage is an assumption supplied by the experimenter, not inferred here.
    """
    if len(risks) < 2 or coverage < 0 or smoothing <= 0:
        raise ValueError("need >=2 risks, nonnegative coverage and positive smoothing")
    c = contrasts(len(risks), device=risks.device, dtype=risks.dtype)
    delta = c.T @ risks
    return risks.mean() + coverage * torch.sqrt(delta.square().sum() + smoothing**2)


def robust_state(model, source, calibration, args, response_scale, response_weight):
    risks = torch.stack([mean_nll(model(e["images"]), e["labels"]) for e in source]).double()
    envelope = risk_envelope(risks, args.coverage, args.envelope_smoothing)
    regularization = args.robust_l2 * sum(p.double().square().sum() for p in model.parameters())
    response_term = risks.new_zeros(())
    if response_weight > 0:
        terms = source_terms(model, calibration)
        identity = torch.eye(len(source)-1, device=risks.device, dtype=torch.float64)
        response_term = response(terms, identity, args.rho)["full_B"] / response_scale
    objective = envelope + regularization + response_weight * response_term
    return {"tensor": objective, "value": float(objective.detach()),
            "source_risks": risks.detach().cpu().numpy(),
            "envelope": float(envelope.detach()),
            "response": float(response_term.detach())}


def run_robust(method, initial, source, target, calibration, args, seed):
    """Joint encoder/head training; no IRM compatibility projection or IRM veto."""
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    response_weight = 0. if method == "robust" else args.robust_response_weight
    response_scale = 1.
    if response_weight > 0:
        with torch.no_grad():
            terms = source_terms(model, calibration)
            identity = torch.eye(len(source)-1, dtype=torch.float64)
            response_scale = max(float(response(terms, identity, args.rho)["full_B"]), 1e-8)
    state_fn = lambda current: robust_state(current, source, calibration, args,
                                            response_scale, response_weight)
    trace, guards = [], []
    start = time.monotonic()
    for step in range(args.steps):
        if step == args.steps-1:
            official_final = evaluate(model, source, target)
        before = state_fn(model)
        if method == "robust_guard":
            previous = vector(model)
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            irm_before = base_state(model, source, 10000.)["value"]
        optimizer.zero_grad(set_to_none=True)
        before["tensor"].backward()
        optimizer.step()
        if method == "robust_guard":
            row = guard_update(model, optimizer, previous, optimizer_before, source,
                               0., args, state_fn=state_fn)
            row["guard_objective"] = "source_extrapolation_plus_response"
            row["irm_before"] = irm_before
            row["irm_after"] = base_state(model, source, 10000.)["value"]
            row["irm_increased"] = row["irm_after"] > irm_before
            guards.append({"step": step} | row)
        if step % args.log_every == 0 or step == args.steps-1:
            after = state_fn(model)
            metrics = evaluate(model, source, target)
            trace.append({"step": step, "objective_before": before["value"],
                          "objective_after": after["value"], "envelope": after["envelope"],
                          "normalized_response": after["response"]} | metrics)
            print(f"{method} seed={seed} step={step} target={metrics['target_accuracy']:.4f}", flush=True)
    return {"algorithm_version": ALGORITHM_VERSION, "method": method, "seed": seed,
            "official_final_pre_update": official_final,
            "final_post_update": evaluate(model, source, target),
            "fixed_scales": {"response_scale": response_scale, "fixed_at_step": 0},
            "guard_status_counts": dict(Counter(r["status"] for r in guards)),
            "accepted_irm_increase": sum(r["applied"] and r["irm_increased"] for r in guards),
            "accepted_all_source_increase": sum(r["applied"] and min(r["full_source_changes"]) > 0 for r in guards),
            "trace": trace, "response_steps": [], "guard_steps": guards,
            "wall_seconds": time.monotonic()-start}


def vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()


def set_vector(model, value):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(value[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def calibration_subset(source, count, seed):
    generator = torch.Generator().manual_seed(seed + 20000)
    return [{k: v[index] for k, v in env.items()}
            for env in source
            for index in [torch.randperm(len(env["labels"]), generator=generator)[:count]]]


def base_state(model, source, weight):
    """Full-data IRM objective; same fixed weight during each guard comparison."""
    logits = [model(e["images"]) for e in source]
    nlls = torch.stack([mean_nll(x, e["labels"]) for x, e in zip(logits, source)])
    penalties = torch.stack([irm_penalty(x, e["labels"]) for x, e in zip(logits, source)])
    objective = (nlls.mean() + .001 * sum(p.norm().square() for p in model.parameters())
                 + weight * penalties.mean()) / max(weight, 1.)
    return {"tensor": objective, "value": float(objective.detach()),
            "source_risks": nlls.detach().cpu().numpy()}


def flat_grad(loss, model):
    return torch.cat([g.detach().flatten() for g in
                      torch.autograd.grad(loss, tuple(model.parameters()))])


def assign_grad(model, flat):
    offset = 0
    for p in model.parameters():
        p.grad = flat[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()


def compatible_response_gradient(base, extra, ratio):
    """Remove conflicts, then cap auxiliary norm relative to the IRM gradient.

    The first-order guarantee applies before Adam preconditioning only.
    The guard separately tests the actually applied finite step.
    """
    if ratio < 0 or not np.isfinite(ratio):
        raise ValueError("gradient ratio must be finite and nonnegative")
    if not torch.isfinite(base).all() or not torch.isfinite(extra).all():
        raise FloatingPointError("nonfinite gradient")
    b, e = base.double(), extra.double()
    norm2 = b @ b
    if float(norm2) <= 1e-24 or ratio == 0:
        return torch.zeros_like(extra)
    e = e - torch.minimum(b @ e, norm2.new_zeros(())) / norm2 * b
    e = e * torch.clamp(ratio * norm2.sqrt() / e.norm().clamp_min(1e-30), max=1.)
    return e.to(extra.dtype)


def guard_update(model, optimizer, previous, optimizer_before, source, weight, args, state_fn=None):
    """Check finite descent of a supplied empirical objective, not target risk.

    Allows every source NLL to rise. No relative head optimum is subtracted.
    Fallback uses the supplied objective gradient and restores Adam state.
    With no state_fn, preserve the legacy v2 full IRM objective.
    """
    if state_fn is None:
        state_fn = lambda current: base_state(current, source, weight)
    proposal = vector(model) - previous
    set_vector(model, previous)
    before = state_fn(model)
    gradient = flat_grad(before["tensor"], model)
    length = min(max(float(proposal.norm()), args.fallback_step), args.max_step_norm)
    fallback = -gradient / max(float(gradient.norm()), 1e-30) * length
    row = {"base_before": before["value"], "candidates": []}
    for kind, direction in (("PROPOSAL", proposal), ("FALLBACK", fallback)):
        if kind == "FALLBACK":
            optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        slope = float(gradient.double() @ direction.double())
        if not np.isfinite(slope) or slope >= 0:
            row["candidates"].append({"direction": kind, "status": "NOT_BASE_DESCENT"})
            continue
        for attempt in range(args.backtracks):
            alpha = 2. ** (-attempt)
            candidate = previous + alpha * direction
            if torch.equal(candidate, previous):
                break
            set_vector(model, candidate)
            actual_slope = float(gradient.double() @ (vector(model)-previous).double())
            after = state_fn(model)
            threshold = before["value"] + args.armijo * actual_slope - args.margin
            accepted = bool(actual_slope < 0 and np.isfinite(after["value"])
                            and after["value"] < before["value"]
                            and after["value"] <= threshold)
            row["candidates"].append({"direction": kind, "alpha": alpha,
                                      "base_after": after["value"], "threshold": threshold,
                                      "accepted": accepted})
            if accepted:
                return row | {
                    "status": kind+"_ACCEPTED", "applied": True, "alpha": alpha,
                    "base_after": after["value"], "threshold": threshold,
                    "full_source_changes": (after["source_risks"]-before["source_risks"]).tolist(),
                    "parameter_change_norm": float((vector(model)-previous).norm()),
                    "optimizer_state_policy": "accepted_adam_moments" if kind == "PROPOSAL" else "restored_preproposal",
                }
            del after
    set_vector(model, previous)
    optimizer.load_state_dict(copy.deepcopy(optimizer_before))
    return row | {"status": "NO_BASE_DESCENT_FOUND", "applied": False, "alpha": None,
                  "base_after": before["value"], "full_source_changes": [0.] * len(source),
                  "parameter_change_norm": 0., "optimizer_state_policy": "restored_preproposal"}


def run(method, initial, source, target, calibration, args, seed):
    if method in ROBUST_METHODS:
        return run_robust(method, initial, source, target, calibration, args, seed)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    gate = torch.eye(len(source)-1, dtype=torch.float64)
    scales = None
    trace, guard_rows, response_rows = [], [], []
    start = time.monotonic()
    for step in range(args.steps):
        active = step >= args.warmup
        if active and method != "irm" and scales is None:
            with torch.no_grad():
                terms = source_terms(model, calibration)
                full = response(terms, gate, args.rho)
                observation_scale = max(float(terms["q"].square().mean().sqrt()), 1e-6)
                omega_ref = max(float(terms["q"].square().mean()), 1e-12)
                response_scale = max(float(full["full_B"]), 1e-12)
                scales = {"observation_scale": observation_scale, "omega_reference": omega_ref,
                          "full_B_reference": response_scale,
                          "gamma_effective": args.response_weight * omega_ref / response_scale,
                          "fixed_at_step": step}
        if active and method != "irm" and (step-args.warmup) % args.gate_every == 0:
            with torch.no_grad():
                gate = soft_gate(source_terms(model, calibration)["q"], scales["observation_scale"], args.tau)
        if step == args.steps-1:
            official_final = evaluate(model, source, target)
        logits = [model(env["images"]) for env in source]
        nlls = [mean_nll(x, env["labels"]) for x, env in zip(logits, source)]
        penalties = [irm_penalty(x, env["labels"]) for x, env in zip(logits, source)]
        weight = 10000. if active else 1.
        objective = (torch.stack(nlls).mean()
                     + .001 * sum(p.norm().square() for p in model.parameters())
                     + weight * torch.stack(penalties).mean())
        if weight > 1:
            objective = objective / weight
        if active and method != "irm" and args.response_weight > 0:
            terms = source_terms(model, calibration)
            resp = response(terms, gate, args.rho)
            extra = resp["full_B"] if method in ("full_alignment", "full_compatible") else resp["B"]
            extra = scales["gamma_effective"] * extra
            base_gradient = flat_grad(objective, model)
            extra_gradient = flat_grad(extra, model)
            correction = (extra_gradient if method == "full_alignment" else
                          compatible_response_gradient(base_gradient, extra_gradient,
                                                       args.max_response_grad_ratio))
            response_rows.append({"step": step, "q_gate": float(gate.item()),
                                  "B": float(resp["B"].detach()), "full_B": float(resp["full_B"].detach()),
                                  "D": float(resp["D"].detach()),
                                  "weighted_extra": float(extra.detach()),
                                  "base_gradient_norm": float(base_gradient.norm()),
                                  "raw_extra_gradient_norm": float(extra_gradient.norm()),
                                  "applied_extra_gradient_norm": float(correction.norm()),
                                  "base_correction_dot": float(base_gradient.double() @ correction.double())})
        guarded = active and method == "proxy_guard"
        if guarded:
            previous = vector(model)
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            full_before = np.array([float(x.detach()) for x in nlls])
        optimizer.zero_grad(set_to_none=True)
        if active and method != "irm" and args.response_weight > 0:
            assign_grad(model, base_gradient + correction)
        else:
            objective.backward()
        optimizer.step()
        if guarded:
            row = guard_update(model, optimizer, previous, optimizer_before, source, weight, args)
            with torch.no_grad():
                full_after = np.array([float(mean_nll(model(e["images"]), e["labels"])) for e in source])
            row["full_source_changes"] = (full_after-full_before).tolist()
            guard_rows.append({"step": step} | row)
        if step % args.log_every == 0 or step == args.steps-1:
            metrics = evaluate(model, source, target)
            trace.append({"step": step, "objective": float(objective.detach()),
                          "q_gate": float(gate.item())} | metrics)
            print(f"{method} seed={seed} step={step} target={metrics['target_accuracy']:.4f} "
                  f"guard={guard_rows[-1]['status'] if guarded else 'off'}", flush=True)
    return {"algorithm_version": ALGORITHM_VERSION, "method": method, "seed": seed, "official_final_pre_update": official_final,
            "final_post_update": evaluate(model, source, target), "fixed_scales": scales,
            "guard_status_counts": dict(Counter(r["status"] for r in guard_rows)),
            "accepted_any_source_increase": sum(r["applied"] and max(r["full_source_changes"])>0 for r in guard_rows),
            "accepted_all_source_increase": sum(r["applied"] and min(r["full_source_changes"])>0 for r in guard_rows),
            "trace": trace, "response_steps": response_rows, "guard_steps": guard_rows,
            "wall_seconds": time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/sunlay/Desktop/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["irm", "robust", "robust_response", "robust_guard"],
                        choices=["irm", "full_alignment", "full_compatible", "blind", "proxy_guard", *ROBUST_METHODS])
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-source", type=int, default=25000)
    parser.add_argument("--n-target", type=int, default=10000)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--rho", type=float, default=1.)
    parser.add_argument("--region-radius", type=float, default=.5, help="Legacy v1 argument; unused")
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--response-weight", type=float, default=1.)
    parser.add_argument("--coverage", type=float, default=1., help="Assumed L2 affine environment radius for robust methods; unused by v2 controls")
    parser.add_argument("--envelope-smoothing", type=float, default=1e-4)
    parser.add_argument("--robust-response-weight", type=float, default=.01)
    parser.add_argument("--robust-l2", type=float, default=.001)
    parser.add_argument("--gate-every", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=200, help="Legacy v1 argument; unused")
    parser.add_argument("--inner-tolerance", type=float, default=1e-6, help="Legacy v1 argument; unused")
    parser.add_argument("--backtracks", type=int, default=16)
    parser.add_argument("--armijo", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.)
    parser.add_argument("--max-response-grad-ratio", type=float, default=.25)
    parser.add_argument("--fallback-step", type=float, default=.001)
    parser.add_argument("--max-step-norm", type=float, default=.1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if (min(args.rho, args.tau) <= 0 or args.margin < 0 or args.warmup < 0
        or min(args.steps,args.hidden_dim,args.calibration_size,args.gate_every,args.backtracks,args.log_every)<1):
        parser.error("positive sizes/scales and nonnegative warmup/margin required")
    if (not 0 < args.armijo < 1 or args.response_weight < 0 or args.max_response_grad_ratio < 0
        or min(args.coverage, args.robust_response_weight, args.robust_l2) < 0
        or args.envelope_smoothing <= 0
        or min(args.fallback_step, args.max_step_norm) <= 0
        or not all(np.isfinite(v) for v in vars(args).values() if isinstance(v, (int, float)))):
        parser.error("invalid finite scales or Armijo parameters")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; preserve historical results using a new run directory")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               "reference_commit": REFERENCE, "scope": "EMPIRICAL_SOURCE_ENVELOPE_WITH_ASSUMED_COVERAGE",
               "algorithm_version": ALGORITHM_VERSION,
               "notes": ["robust methods optimize an affine-environment risk envelope, not IRM; no IRM warmup or veto.",
                         "coverage is an explicit assumption; source data alone do not certify target coverage.",
                         "robust_response/robust_guard add normalized full gradient/Hessian response; robust is the necessary no-response control.",
                         "Legacy v2 methods retain their original IRM objective and update rules.",
                         "blind/proxy_guard project conflicting auxiliary gradients and cap their norm; full_alignment remains a raw control.",
                         "No target-risk certificate or OOD improvement is claimed.",
                         "E=2 blind response is scalar reweighting, not directional selection.",
                         "region-radius, inner-steps, inner-tolerance are legacy unused arguments. coverage is active only for robust methods.",
                         "Accepted scaled Adam proposals retain proposed moments; fallback/rejection restore preproposal state."],
               "fingerprints": {}, "results": []}
    (args.output/"configuration.json").write_text(json.dumps(payload,indent=2)+"\n")
    for seed in args.seeds:
        source, target, fingerprint = build_official_envs(args.data_root,seed,args.n_source,args.n_target,"cpu")
        initial = MLP(args.hidden_dim)
        calibration = calibration_subset(source,args.calibration_size,seed)
        payload["fingerprints"][str(seed)] = fingerprint
        for method in args.methods:
            result = run(method,initial,source,target,calibration,args,seed)
            payload["results"].append(result)
            (args.output/f"{method}_seed{seed}.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
            (args.output/"all_results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
            print(json.dumps({k:v for k,v in result.items() if k not in ("trace","response_steps","guard_steps")}),flush=True)


if __name__ == "__main__":
    main()
