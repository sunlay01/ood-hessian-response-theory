import unittest

import numpy as np
import torch
from scipy.optimize import minimize

from response_certificate_core import (contrasts, head_tensor, response,
                                       soft_gate, solve_head_opt, source_terms)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(),
                                        torch.nn.Linear(3, 3), torch.nn.Tanh(),
                                        torch.nn.Linear(3, 1)).double()


class ResponseCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TinyModel()
        self.envs = [dict(images=torch.randn(9, 2).double() + e,
                          labels=torch.randint(0, 2, (9, 1)).double()) for e in range(2)]

    def test_helmert(self):
        c = contrasts(5)
        torch.testing.assert_close(c.T @ c, torch.eye(4).double())
        torch.testing.assert_close(c.T @ torch.ones(5).double(), torch.zeros(4).double())

    def test_analytic_head_derivatives_and_bias_scaling(self):
        terms = source_terms(self.model, self.envs)
        w = head_tensor(self.model).detach().requires_grad_()
        for i, (x, y) in enumerate(zip(terms['xs'], terms['ys'])):
            def loss(v):
                return torch.nn.functional.binary_cross_entropy_with_logits(x @ v, y)
            g = torch.autograd.functional.jacobian(loss, w)
            h = torch.autograd.functional.hessian(loss, w)
            torch.testing.assert_close(g, terms['gradients'][i])
            torch.testing.assert_close(h, terms['hessians'][i])
            scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            q = torch.autograd.grad(loss(w * scale), scale)[0]
            torch.testing.assert_close(q, terms['q'][i])

    def test_hessian_response_encoder_gradient_finite_difference(self):
        q = torch.tensor([[.37]], dtype=torch.float64)
        def penalty():
            terms = source_terms(self.model, self.envs)
            # Isolate curvature to catch accidental detachment of H.
            return (response(terms, q, 1.)['GH'] @ q).square().sum()
        p = self.model.main[0].weight
        gradient = torch.autograd.grad(penalty(), p)[0]
        self.assertGreater(float(gradient.norm()), 1e-9)
        original, eps = float(p[0, 0].detach()), 1e-5
        with torch.no_grad():
            p[0, 0] = original + eps
        plus = float(penalty().detach())
        with torch.no_grad():
            p[0, 0] = original - eps
        minus = float(penalty().detach())
        with torch.no_grad():
            p[0, 0] = original
        self.assertAlmostEqual(float(gradient[0, 0]), (plus - minus) / (2 * eps), places=8)

    def test_soft_gate_detached_fixed_scale_and_e2_full_response(self):
        terms = source_terms(self.model, self.envs)
        gate = soft_gate(terms['q'], .2, .3)
        self.assertFalse(gate.requires_grad)
        gate_small = soft_gate(terms['q'] * .01, .2, .3)
        self.assertGreater(float(gate_small), float(gate))
        values = [response(terms, torch.tensor([[v]], dtype=torch.float64), 1.)['D']
                  for v in (0., .1, .5, 1.)]
        for value in values[1:]:
            torch.testing.assert_close(value, values[0])

    def test_convex_gap_brackets_reference_even_without_convergence(self):
        rng = np.random.RandomState(8)
        xs = [np.column_stack((rng.randn(n, 2), np.ones(n))) for n in (12, 19)]
        ys = [rng.randint(0, 2, len(x)) for x in xs]
        center, radius = np.array([.4, -.1, .2]), .7
        def objective(u):
            return np.mean([np.mean(np.logaddexp(0, x @ u) - y * (x @ u))
                            for x, y in zip(xs, ys)])
        reference = minimize(objective, center, method='SLSQP',
                             constraints=[dict(type='ineq', fun=lambda u: radius ** 2 - np.sum((u - center) ** 2))],
                             options=dict(ftol=1e-13, maxiter=500))
        self.assertTrue(reference.success)
        for budget in (0, 1, 150):
            result = solve_head_opt(xs, ys, center, radius, max_iter=budget, tol=1e-8)
            self.assertLessEqual(np.linalg.norm(result['head'] - center), radius + 1e-12)
            self.assertLessEqual(result['risk'] - result['gap'], reference.fun + 1e-9)
            self.assertGreaterEqual(result['risk'], reference.fun - 1e-9)
        self.assertLess(result['gap'], 1e-6)

    def test_changed_representation_reoptimizes_head(self):
        x = np.column_stack((np.linspace(-2, 2, 21), np.ones(21)))
        y = (x[:, 0] > .2).astype(float)
        initial = solve_head_opt([x], [y], np.zeros(2), 2., max_iter=500)
        changed = x.copy()
        changed[:, 0] *= -1
        updated = solve_head_opt([changed], [y], np.zeros(2), 2., start=initial['head'], max_iter=500)
        self.assertLess(initial['head'][0] * updated['head'][0], 0)
        self.assertAlmostEqual(initial['risk'], updated['risk'], places=5)


if __name__ == '__main__':
    unittest.main()
