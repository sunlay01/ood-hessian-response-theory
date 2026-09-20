"""IRMv1 reference-protocol CMNIST with a certified finite-step head repair.

Reference: facebookresearch/InvariantRiskMinimization, commit
fc185d0f828a98f57030ba3647efc7394d1be95a, code/colored_mnist/main.py.

The encoder takes Adam's proposed step. Its finite source-risk offset is
measured exactly on the two empirical environments. A small convex head
problem accounts for that offset and the global logistic curvature bound.
Uncertified proposals restore ALL parameters. Adam moments advance on each
attempt (explicit projected-Adam policy). Targets never enter selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from run_cmnist_irm_calibration import MLP, load_mnist, make_environment, mean_nll, mean_accuracy

REFERENCE = "fc185d0f828a98f57030ba3647efc7394d1be95a"


def official_indices(seed, n_source=25000, n_target=10000):
    if not 0 < n_source <= 25000 or not 0 < n_target <= 10000:
        raise ValueError("invalid official split sizes")
    order = np.random.RandomState(seed).permutation(50000)
    return (torch.as_tensor(order[::2][:n_source]),
            torch.as_tensor(order[1::2][:n_source]),
            torch.arange(50000, 50000 + n_target))


def build_official_envs(root, seed, n_source, n_target, device):
    images, labels, _, _ = load_mnist(root)
    indices = official_indices(seed, n_source, n_target)
    # Official code draws source and validation noise from a single torch RNG
    # before initializing the model. Resetting it once reproduces that order.
    torch.manual_seed(seed)
    envs = [
        make_environment(images, labels, idx, flip, torch.default_generator, .25)
        for idx, flip in zip(indices, (.2, .1, .9))
    ]
    fingerprint = hashlib.sha256()
    for env in envs:
        for value in env.values():
            fingerprint.update(value.numpy().tobytes())
    return (
        [{k: v.to(device) for k, v in e.items()} for e in envs[:2]],
        {k: v.to(device) for k, v in envs[2].items()},
        fingerprint.hexdigest(),
    )


def forward_features(model, images):
    features = model.main[:4](images.flatten(1))
    return features, model.main[4](features)


def irm_penalty(logits, labels):
    scale = logits.new_ones((), requires_grad=True)
    grad = torch.autograd.grad(mean_nll(logits * scale, labels), scale, create_graph=True)[0]
    return grad.square()


def head_vector(model):
    layer = model.main[4]
    return torch.cat((layer.weight.detach().flatten(), layer.bias.detach())).cpu().double().numpy()


def set_head(model, value):
    layer = model.main[4]
    value = torch.as_tensor(value, dtype=layer.weight.dtype, device=layer.weight.device)
    with torch.no_grad():
        layer.weight.copy_(value[:-1].reshape_as(layer.weight))
        layer.bias.copy_(value[-1:].reshape_as(layer.bias))


def design(features):
    x = features.detach().cpu().double().numpy()
    return np.column_stack((x, np.ones(len(x))))


def logistic_risk(x, y, head):
    logits = x @ head
    return float(np.mean(np.logaddexp(0., logits) - y * logits))


def solve_head_repair(xs, ys, before, old_head, proposed_head, rho):
    """Certify total source decrease, including the encoder's finite offset.

    v = old_head - repaired_head; in a frozen design X the logistic upper
    bound is R(old-v)-R(old) <= -g.v + mean((Xv)^2)/8.
    Search in span{Adam head step, g_source0, g_source1}. Failure in this
    subspace is not proof that full-network repair is impossible.
    """
    gradients, offsets = [], []
    for x, y, reference in zip(xs, ys, before):
        logits = x @ old_head
        probs = np.exp(-np.logaddexp(0., -logits))
        gradients.append(x.T @ (probs - y) / len(y))
        offsets.append(logistic_risk(x, y, old_head) - reference)
    gradients, offsets = np.asarray(gradients), np.asarray(offsets)
    base = old_head - proposed_head

    def certificates(v):
        return np.array([
            g @ v - np.mean((x @ v)**2)/8. - offset
            for x, g, offset in zip(xs, gradients, offsets)
        ])

    base_cert = certificates(base)
    common = {
        "encoder_risk_offsets": offsets.tolist(),
        "base_certificate_min": float(base_cert.min()),
        "rho": rho,
    }
    if float(base_cert.min()) >= rho:
        return common | {"status": "BASE_CERTIFIED", "head": proposed_head, "attempts": []}

    u, s, _ = np.linalg.svd(np.column_stack((base, gradients.T)), full_matrices=False)
    rank = int(np.sum(s > (s[0] * 1e-12))) if s[0] > 0 else 0
    if rank == 0:
        return common | {"status": "NO_SEARCH_DIRECTION", "head": None, "attempts": []}
    basis = u[:, :rank]
    projected = [x @ basis for x in xs]
    # Q_e bounds the Hessian over the entire head step, not at one point.
    qs = np.array([z.T @ z / (4. * len(z)) for z in projected])
    a = gradients @ basis
    base_z = basis.T @ base
    scale = max(float(np.linalg.norm(base_z)), 1e-3)
    constraints_scale = max(rho, 1e-8)

    def margin(z):
        return a @ z - .5*np.einsum("i,eij,j->e", z, qs, z) - offsets

    def cons(t):
        return (margin(t * scale) - rho * 1.001) / constraints_scale

    def jac(t):
        return (a - np.einsum("eij,j->ei", qs, t*scale)) * scale / constraints_scale

    attempts, accepted = [], []
    for initial in (base_z/scale, np.zeros(rank)):
        result = minimize(
            lambda t: .5*np.sum((t-base_z/scale)**2), initial,
            jac=lambda t: t-base_z/scale, method="SLSQP",
            constraints={"type": "ineq", "fun": cons, "jac": jac},
            options={"maxiter": 100, "ftol": 1e-10},
        )
        v = basis @ (result.x * scale)
        # Model parameters are float32: certify the ACTUAL rounded head.
        head = (old_head - v).astype(np.float32).astype(np.float64)
        cert = certificates(old_head - head)
        valid = bool(np.all(np.isfinite(head)) and np.all(cert >= rho))
        attempts.append({
            "solver_success": bool(result.success), "solver_status": int(result.status),
            "message": str(result.message), "verified": valid,
            "certificate_min": float(cert.min()) if np.all(np.isfinite(cert)) else None,
        })
        if valid:
            accepted.append((float(np.linalg.norm(head-proposed_head)), head))
    if not accepted:
        return common | {"status": "NO_CERTIFIED_REPAIR", "head": None, "attempts": attempts}
    _, head = min(accepted, key=lambda x: x[0])
    return common | {"status": "REPAIRED", "head": head, "attempts": attempts}


def apply_guard(model, snapshot, old_head, before, source, rho):
    """Model already contains the Adam proposal; restore it on rejection."""
    proposed_head = head_vector(model)
    with torch.no_grad():
        xs = [design(forward_features(model, e["images"])[0]) for e in source]
    ys = [e["labels"].detach().cpu().double().numpy().reshape(-1) for e in source]
    result = solve_head_repair(xs, ys, before, old_head, proposed_head, rho)
    selected = result.pop("head")
    if selected is None:
        with torch.no_grad():
            for p, previous in zip(model.parameters(), snapshot):
                p.copy_(previous)
        return result | {
            "applied": False, "head_correction_norm": None,
            "actual_source_changes": None, "certificate_min": None,
            "bound_violation": None,
        }
    set_head(model, selected)
    actual_head = head_vector(model)
    v = old_head - actual_head
    certs, changes, bounds = [], [], []
    for x, y, reference in zip(xs, ys, before):
        mixed = logistic_risk(x, y, old_head)
        probs = np.exp(-np.logaddexp(0., -(x @ old_head)))
        g = x.T @ (probs-y) / len(y)
        certificate = float(g@v - np.mean((x@v)**2)/8. - (mixed-reference))
        certs.append(certificate)
        changes.append(logistic_risk(x, y, actual_head)-reference)
        bounds.append(-certificate)
    violation = bool(np.any(np.array(changes) > np.array(bounds) + 1e-10))
    if min(certs) < rho or violation:
        raise AssertionError("accepted update fails independent finite-step audit")
    return result | {
        "applied": True,
        "head_correction_norm": float(np.linalg.norm(actual_head-proposed_head)),
        "actual_source_changes": changes, "certificate_min": min(certs),
        "descent_bounds": bounds, "bound_violation": violation,
    }


def evaluate(model, source, target):
    with torch.no_grad():
        source_logits = [model(e["images"]) for e in source]
        target_logits = model(target["images"])
        return {
            "source_accuracy": float(np.mean([mean_accuracy(x, e["labels"]) for x, e in zip(source_logits, source)])),
            "source_nll": float(np.mean([float(mean_nll(x, e["labels"])) for x, e in zip(source_logits, source)])),
            "target_accuracy": mean_accuracy(target_logits, target["labels"]),
            "target_nll": float(mean_nll(target_logits, target["labels"])),
        }


def run_method(method, initial, source, target, args, seed):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trace, guard_rows = [], []
    started = time.time()
    official_final = None
    for step in range(args.steps):
        features_and_logits = [forward_features(model, e["images"]) for e in source]
        nlls = [mean_nll(pair[1], e["labels"]) for pair, e in zip(features_and_logits, source)]
        penalties = [irm_penalty(pair[1], e["labels"]) for pair, e in zip(features_and_logits, source)]
        weight = 10000. if step >= 100 else 1.
        objective = (torch.stack(nlls).mean()
                     + .001 * sum(p.norm().square() for p in model.parameters())
                     + weight * torch.stack(penalties).mean())
        if weight > 1:
            objective = objective / weight
        # The official script reports the final loop's PRE-update metric.
        if step == args.steps-1:
            official_final = evaluate(model, source, target)
        repair_end = getattr(args, "repair_end", None)
        guarded = (method == "irm_signed" and step >= args.repair_start
                   and (repair_end is None or step < repair_end))
        if guarded:
            old_head = head_vector(model)
            before = [
                logistic_risk(design(pair[0]), e["labels"].cpu().double().numpy().reshape(-1), old_head)
                for pair, e in zip(features_and_logits, source)
            ]
            snapshot = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if guarded:
            # Keep Adam's moments advanced even when parameter update is rejected.
            # Otherwise the same rejected full-batch proposal repeats forever.
            guard_rows.append({"step": step} | apply_guard(
                model, snapshot, old_head, before, source, args.rho))
        if step % args.log_every == 0 or step == args.steps-1:
            metrics = evaluate(model, source, target)
            trace.append({"step": step, "penalty_weight": weight,
                          "objective": float(objective.detach())} | metrics)
            print(f"{method} seed={seed} step={step} target={metrics['target_accuracy']:.4f}"
                  f" guard={guard_rows[-1]['status'] if guarded else 'off'}", flush=True)
        del features_and_logits, nlls, penalties, objective
    statuses = {}
    for row in guard_rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "method": method, "seed": seed, "official_final_pre_update": official_final,
        "final_post_update": evaluate(model, source, target),
        "guard_status_counts": statuses,
        "accepted_source_all_decrease": sum(
            r["applied"] and max(r["actual_source_changes"]) < 0 for r in guard_rows),
        "bound_violations": sum(r["bound_violation"] is True for r in guard_rows),
        "trace": trace, "guard_steps": guard_rows,
        "wall_seconds": time.time()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/sunlay/Desktop/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=("irm", "irm_signed"), default=["irm", "irm_signed"])
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-source", type=int, default=25000)
    parser.add_argument("--n-target", type=int, default=10000)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--repair-start", type=int, default=0)
    parser.add_argument("--repair-end", type=int, default=None,
                        help="Exclusive guard end step; diagnostic ablation only.")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.rho <= 0 or not np.isfinite(args.rho) or args.steps < 1 or args.log_every < 1:
        parser.error("rho, steps, log-every must be positive")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {
        "configuration": config, "reference_commit": REFERENCE,
        "official_protocol_sizes": args.n_source == 25000 and args.n_target == 10000 and args.hidden_dim == 256 and args.steps == 501,
        "protocol_notes": {
            "validation_split": "MNIST train[50000:60000], NOT t10k",
            "source_split": "shuffled train[:50000], alternating examples",
            "label_noise": .25, "source_color_flips": [.2, .1], "validation_color_flip": .9,
            "optimizer": "Adam lr=.001; L2=.001; 1->10000 at step100; whole-loss rescaling; no Adam reset",
            "reporting": "both official pre-final-update and post-final-update metrics",
            "restarts": "seeded paired local run; official default is 10 restarts",
            "repair_scope": "source empirical risk; frozen proposed encoder + convex head repair in at most 3 dimensions",
            "moment_policy": "Adam moments advance on rejected proposals; all parameters restored",
            "certificate_precision": "float64 logistic risks on computed features; actual head rounded to model float32 before certification",
            "target_used_for_selection": False,
        }, "data_fingerprints": {}, "results": [],
    }
    (args.output/"configuration.json").write_text(json.dumps(payload, indent=2)+"\n")
    for seed in args.seeds:
        source, target, fingerprint = build_official_envs(
            args.data_root, seed, args.n_source, args.n_target, args.device)
        initial = MLP(args.hidden_dim).to(args.device)
        payload["data_fingerprints"][str(seed)] = fingerprint
        for method in args.methods:
            result = run_method(method, initial, source, target, args, seed)
            payload["results"].append(result)
            (args.output/f"{method}_seed{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            (args.output/"all_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ("trace", "guard_steps")}), flush=True)


if __name__ == "__main__":
    main()
