"""Differentiable finite source contrasts and bounded empirical head optimum.

These quantities describe observed source contrasts, not an unidentified
environment derivative or a guarantee for an arbitrary unseen target.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit


def head_tensor(model):
    layer = model.main[4]
    return torch.cat((layer.weight.reshape(-1), layer.bias.reshape(-1))).double()


def contrasts(e, *, device=None, dtype=torch.float64):
    if e < 2:
        raise ValueError("at least two source environments are required")
    c = torch.zeros((e, e - 1), device=device, dtype=dtype)
    for j in range(e - 1):
        denom = ((j + 1) * (j + 2)) ** .5
        c[:j + 1, j] = 1 / denom
        c[j + 1, j] = -(j + 1) / denom
    return c


def source_terms(model, envs):
    head = head_tensor(model)
    xs, ys, risks, qs, gradients, hessians = [], [], [], [], [], []
    for env in envs:
        features = model.main[:4](env["images"].flatten(1)).double()
        x = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
        y = env["labels"].reshape(-1).double()
        logits = x @ head
        p = torch.sigmoid(logits)
        g = x.T @ (p - y) / len(y)
        h = x.T @ ((p * (1 - p))[:, None] * x) / len(y)
        xs.append(x)
        ys.append(y)
        risks.append(F.binary_cross_entropy_with_logits(logits, y))
        qs.append(g @ head)  # Scaling the complete logit includes head bias.
        gradients.append(g)
        hessians.append(h)
    return dict(risks=torch.stack(risks), q=torch.stack(qs),
                gradients=torch.stack(gradients), hessians=torch.stack(hessians),
                xs=xs, ys=ys)


def soft_gate(q, scale, tau):
    if float(scale) <= 0 or tau <= 0:
        raise ValueError("fixed calibration scale and tau must be positive")
    q = q.detach()
    c = contrasts(len(q), device=q.device, dtype=q.dtype)
    o = (q.reshape(1, -1) / scale) @ c
    identity = torch.eye(c.shape[1], dtype=q.dtype, device=q.device)
    return torch.linalg.solve(o.T @ o + tau ** 2 * identity,
                              tau ** 2 * identity).detach()


def response(terms, Q, rho):
    if rho <= 0:
        raise ValueError("probe radius must be positive")
    g, h = terms["gradients"], terms["hessians"]
    c = contrasts(len(g), device=g.device, dtype=g.dtype)
    gg = g.T @ c
    gh = h.flatten(1).T @ c
    Q = Q.detach().to(device=g.device, dtype=g.dtype)
    visible = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype) - Q
    bg, bh = gg @ Q, gh @ Q
    return dict(B=4 * rho ** 2 * bg.square().sum() + rho ** 4 * bh.square().sum(),
                full_B=4 * rho ** 2 * gg.square().sum() + rho ** 4 * gh.square().sum(),
                D=2 * rho * (torch.linalg.vector_norm(gg @ visible) + torch.linalg.vector_norm(bg))
                  + rho ** 2 * (torch.linalg.vector_norm(gh @ visible) + torch.linalg.vector_norm(bh)),
                Gg=gg, GH=gh)


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def solve_head_opt(xs, ys, center, radius, start=None, max_iter=150, tol=1e-5):
    """Return feasible primal optimum approximation and certified convex gap.

    The optimum lies in [risk-gap, risk], regardless of convergence. Equal
    environment weights match mean source risk even for unequal sample counts.
    The ball center/radius are fixed within a comparison, not refitted per point.
    """
    xs, ys = [_numpy(x) for x in xs], [_numpy(y).reshape(-1) for y in ys]
    center = _numpy(center).reshape(-1)
    if radius < 0 or max_iter < 0 or tol < 0 or len(xs) != len(ys) or not xs:
        raise ValueError("invalid head optimization arguments")

    def project(u):
        delta = u - center
        norm = np.linalg.norm(delta)
        return center + delta * min(1., radius / max(norm, np.finfo(float).tiny))

    def value_grad(u):
        risk, gradient = 0., np.zeros_like(center)
        for x, y in zip(xs, ys):
            logits = x @ u
            risk += np.mean(np.logaddexp(0., logits) - y * logits)
            gradient += x.T @ (expit(logits) - y) / len(y)
        return float(risk / len(xs)), gradient / len(xs)

    gram = sum(x.T @ x / len(x) for x in xs) / (4 * len(xs))
    lipschitz = float(np.max(np.sum(np.abs(gram), axis=1)))
    u = project(center if start is None else _numpy(start).reshape(-1))
    best_risk, best_grad = value_grad(u)
    best, momentum, t = u.copy(), u.copy(), 1.
    iterations = 0
    for iteration in range(max_iter):
        gap = max(0., float(best_grad @ (best - center) + radius * np.linalg.norm(best_grad)))
        if gap <= tol or lipschitz == 0:
            break
        _, grad = value_grad(momentum)
        candidate = project(momentum - grad / lipschitz)
        risk, gradient = value_grad(candidate)
        # Monotone restart keeps acceleration from increasing the best primal.
        if risk > best_risk:
            momentum, t = best.copy(), 1.
            candidate = project(best - best_grad / lipschitz)
            risk, gradient = value_grad(candidate)
        if risk <= best_risk:
            best, best_risk, best_grad = candidate.copy(), risk, gradient.copy()
        next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = candidate + ((t - 1) / next_t) * (candidate - u)
        u, t = candidate, next_t
        iterations = iteration + 1
    gap = max(0., float(best_grad @ (best - center) + radius * np.linalg.norm(best_grad)))
    return dict(head=best, risk=best_risk, gap=gap, iterations=iterations,
                converged=bool(gap <= tol))
