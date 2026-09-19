"""Numerical algebra check for theory/13.

This is not a proof and does not replace the Markdown derivations.  It checks
the two finite-dimensional observation kernels, the visible/blind constants,
and the displayed target-risk gaps.
"""

import numpy as np


def check_irm():
    a = 0.3
    h = np.array([1.0, -1.0])
    theta = h.copy()
    O = np.array([[1.0, 1.0], [1.0, 1.0]])
    P_perp = np.eye(2) - np.linalg.pinv(O) @ O
    assert np.allclose(O @ h, 0.0)
    assert np.allclose(P_perp @ h, h)
    for eta in (a * h, -a * h):
        q = theta @ theta - 2.0 + theta @ eta + 2.0 * eta[1]
        assert abs(q) < 1e-12
    eps = 0.1
    gap = (1.0 + eps) ** 2 - 1.0
    assert np.allclose(gap, 2.0 * eps + eps**2)
    return float(np.linalg.norm(np.linalg.pinv(O))), float(np.linalg.norm(P_perp)), gap


def check_vrex():
    h = np.array([0.0, 1.0])
    O = np.array([[1.0, 0.0]])
    P_perp = np.eye(2) - np.linalg.pinv(O) @ O
    assert np.allclose(O @ h, 0.0)
    assert np.allclose(P_perp @ h, h)
    eps = 0.1
    gap = 0.5 * (1.0 + eps) ** 2 - 0.5
    assert np.allclose(gap, eps + 0.5 * eps**2)
    # The V-REx residual is first-order blind and second-order visible.
    assert np.allclose(eps * h[0] + 0.5 * np.dot(eps * h, eps * h), 0.5 * eps**2)
    return float(np.linalg.norm(np.linalg.pinv(O))), float(np.linalg.norm(P_perp)), gap


if __name__ == "__main__":
    print("IRMv1  kappa beta gap:", check_irm())
    print("V-REx   kappa beta gap:", check_vrex())
