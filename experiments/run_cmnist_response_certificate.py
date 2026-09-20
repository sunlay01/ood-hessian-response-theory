"""Official-protocol IRMv1 with full/soft-blind response and empirical U guard.

This is an empirical source-contrast experiment, NOT a target-risk certifier.
The full-data IRM loss is unchanged. Response estimation and head optimization
use the same fixed source-only calibration examples for all paired methods.
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
    evaluate, head_vector, irm_penalty, mean_nll)
from response_certificate_core import source_terms, response, soft_gate, solve_head_opt


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


def proxy_state(model, calibration, gate, center, args, initial_head=None):
    terms = source_terms(model, calibration)
    resp = response(terms, gate, args.rho)
    optimum = solve_head_opt(terms["xs"], terms["ys"], center,
                            args.region_radius, start=initial_head,
                            max_iter=args.inner_steps, tol=args.inner_tolerance)
    fixed_head = torch.as_tensor(optimum["head"], dtype=torch.float64)
    opt_risk = torch.stack([
        torch.nn.functional.binary_cross_entropy_with_logits(x @ fixed_head, y.reshape(-1))
        for x, y in zip(terms["xs"], terms["ys"])
    ]).mean()
    # Envelope derivative at an approximate constrained head optimizer.
    # The comparison itself uses convex primal/dual gap intervals below.
    proxy = terms["risks"].mean() - opt_risk + args.coverage * resp["D"]
    lower = float(proxy.detach())
    upper = lower + optimum["gap"]
    return {
        "tensor": proxy, "lower": lower, "upper": upper,
        "source_risks": terms["risks"].detach().tolist(),
        "head_excess_lower": float((terms["risks"].mean()-opt_risk).detach()),
        "D": float(resp["D"].detach()), "B": float(resp["B"].detach()),
        "full_B": float(resp["full_B"].detach()), "optimum": optimum,
    }


def guard_update(model, optimizer, previous, optimizer_before, calibration, gate, args):
    proposed = vector(model)
    delta = proposed - previous
    set_vector(model, previous)
    center = head_vector(model).copy()
    before = proxy_state(model, calibration, gate, center, args, initial_head=center)
    fallback_grads = torch.autograd.grad(before["tensor"], tuple(model.parameters()))
    fallback = -torch.cat([g.detach().flatten() for g in fallback_grads])
    candidates = []
    row = {"proxy_before_lower": before["lower"], "proxy_before_upper": before["upper"],
           "head_gap_before": before["optimum"]["gap"],
           "inner_converged_before": before["optimum"]["converged"],
           "D_before": before["D"], "head_excess_before_lower": before["head_excess_lower"],
           "q_gate": float(gate.item()), "candidates": candidates}
    for kind, direction in (("PROPOSAL", delta), ("FALLBACK", fallback)):
        if kind == "FALLBACK":
            # Rejecting the Adam proposal restores its ENTIRE state, even if
            # a separate gradient-U update is subsequently accepted.
            optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        norm2 = float(direction.double().square().sum())
        for attempt in range(args.backtracks):
            alpha = 2. ** (-attempt)
            set_vector(model, previous + alpha * direction)
            distance = float(np.linalg.norm(head_vector(model)-center))
            if distance > args.region_radius:
                candidates.append({"direction": kind, "alpha": alpha, "status": "OUTSIDE_FIXED_REGION"})
                continue
            with torch.no_grad():
                after = proxy_state(model, calibration, gate, center, args,
                                    initial_head=before["optimum"]["head"])
            armijo = args.armijo * alpha * norm2
            threshold = before["lower"] - armijo - args.margin
            accepted = bool(np.isfinite(after["upper"]) and after["upper"] <= threshold)
            candidates.append({"direction": kind, "alpha": alpha,
                               "proxy_lower": after["lower"], "proxy_upper": after["upper"],
                               "head_gap": after["optimum"]["gap"],
                               "inner_converged": after["optimum"]["converged"],
                               "threshold": threshold, "accepted": accepted})
            if accepted:
                return row | {
                    "status": kind+"_ACCEPTED", "applied": True, "alpha": alpha,
                    "proxy_after_lower": after["lower"], "proxy_after_upper": after["upper"],
                    "armijo_decrease": armijo, "comparison_margin": args.margin,
                    "source_calibration_changes": (np.array(after["source_risks"])-before["source_risks"]).tolist(),
                    "D_after": after["D"], "head_excess_after_lower": after["head_excess_lower"],
                    "parameter_change_norm": float(torch.linalg.vector_norm(vector(model)-previous)),
                    "optimizer_state_policy": "accepted_adam_moments" if kind == "PROPOSAL" else "restored_preproposal",
                }
    set_vector(model, previous)
    optimizer.load_state_dict(copy.deepcopy(optimizer_before))
    return row | {"status": "NO_PROXY_DESCENT_FOUND", "applied": False, "alpha": None,
                  "proxy_after_lower": None, "proxy_after_upper": None,
                  "source_calibration_changes": None, "parameter_change_norm": 0.,
                  "optimizer_state_policy": "restored_preproposal"}


def run(method, initial, source, target, calibration, args, seed):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    gate = torch.ones((1, 1), dtype=torch.float64)
    scales = None
    trace, guard_rows, response_rows = [], [], []
    start = time.monotonic()
    for step in range(args.steps):
        active = step >= args.warmup
        if active and scales is None:
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
        if active and (step-args.warmup) % args.gate_every == 0:
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
        if active and method != "irm":
            terms = source_terms(model, calibration)
            resp = response(terms, gate, args.rho)
            extra = resp["full_B"] if method == "full_alignment" else resp["B"]
            objective = objective + scales["gamma_effective"] * extra
            response_rows.append({"step": step, "q_gate": float(gate.item()),
                                  "B": float(resp["B"].detach()), "full_B": float(resp["full_B"].detach()),
                                  "D": float(resp["D"].detach()),
                                  "weighted_extra": float((scales["gamma_effective"]*extra).detach())})
        guarded = active and method == "proxy_guard"
        if guarded:
            previous = vector(model)
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            full_before = np.array([float(x.detach()) for x in nlls])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if guarded:
            row = guard_update(model, optimizer, previous, optimizer_before, calibration, gate, args)
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
    return {"method": method, "seed": seed, "official_final_pre_update": official_final,
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
    parser.add_argument("--methods", nargs="+", default=["irm", "full_alignment", "blind", "proxy_guard"],
                        choices=["irm", "full_alignment", "blind", "proxy_guard"])
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-source", type=int, default=25000)
    parser.add_argument("--n-target", type=int, default=10000)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--rho", type=float, default=1.)
    parser.add_argument("--region-radius", type=float, default=.5)
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--response-weight", type=float, default=1.)
    parser.add_argument("--coverage", type=float, default=1.)
    parser.add_argument("--gate-every", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--inner-tolerance", type=float, default=1e-6)
    parser.add_argument("--backtracks", type=int, default=8)
    parser.add_argument("--armijo", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1e-7)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if (min(args.rho, args.region_radius, args.tau, args.coverage, args.inner_tolerance) <= 0
        or args.region_radius*2 > args.rho or args.margin < 0
        or min(args.steps,args.calibration_size,args.gate_every,args.inner_steps,args.backtracks,args.log_every)<1):
        parser.error("positive parameters required; 2*region-radius must be <=rho")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               "reference_commit": REFERENCE, "scope": "EMPIRICAL_PROXY_ONLY",
               "notes": ["Full-data official IRMv1 loss; fixed source-only subset for response and U.",
                         "Gamma is a fixed ball around the pre-step head for each comparison; optimum recomputed for each encoder.",
                         "U excludes unknown structural/unseen error; coverage=1 is an assumed proxy coefficient, not validated target coverage.",
                         "Head primal-dual gaps enter acceptance intervals; margin is numerical, not a statistical confidence bound.",
                         "Q is detached and refreshed every 10 steps by default; observation scale fixed at warmup end.",
                         "Both response coefficients use the same source-only warmup full_B calibration.",
                         "E=2 makes D independent of Q; blind_B is Q^2 times full_B.",
                         "Same outer steps, not matched compute; guard head solves and backtracking are additional work.",
                         "Accepted backtracked Adam steps retain proposed moments; fallback or final rejection restores original moments."],
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
