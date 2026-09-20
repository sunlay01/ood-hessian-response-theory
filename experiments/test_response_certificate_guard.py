import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import run_cmnist_response_certificate as m


class GuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = m.MLP(4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        self.optimizer.step()
        self.old = m.vector(self.model)
        self.state = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.step()
        m.set_vector(self.model, self.old*1.1)
        self.args = SimpleNamespace(region_radius=100., backtracks=4, armijo=1e-4, margin=0., fallback_step=.01, max_step_norm=1.)

    def quadratic_base(self, model, *args, **kwargs):
        u = sum(p.double().square().sum() for p in model.parameters())/2
        value = float(u.detach())
        return {"tensor": u, "value": value, "source_risks": np.array([-value, -value])}

    def assert_optimizer_restored(self):
        actual = self.optimizer.state_dict()
        self.assertEqual(actual["param_groups"],self.state["param_groups"])
        for key, state in self.state["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value,actual["state"][key][name]))

    def test_rejection_restores_parameters_and_adam_state(self):
        self.args.margin = 1e9
        with patch.object(m,"base_state",side_effect=self.quadratic_base):
            row = m.guard_update(self.model,self.optimizer,self.old,self.state,[],10000.,self.args)
        self.assertEqual(row["status"],"NO_BASE_DESCENT_FOUND")
        self.assertTrue(torch.equal(m.vector(self.model),self.old))
        self.assert_optimizer_restored()

    def test_fallback_accepts_base_descent_even_when_all_source_risks_rise(self):
        with patch.object(m,"base_state",side_effect=self.quadratic_base):
            row = m.guard_update(self.model,self.optimizer,self.old,self.state,[],10000.,self.args)
        self.assertEqual(row["status"],"FALLBACK_ACCEPTED")
        self.assertGreater(min(row["full_source_changes"]),0)
        self.assertLess(row["base_after"],row["base_before"])
        self.assert_optimizer_restored()

    def test_conflicting_response_is_removed_and_norm_is_capped(self):
        b = torch.tensor([1., 0.])
        e = torch.tensor([-4., 3.])
        c = m.compatible_response_gradient(b, e, .25)
        torch.testing.assert_close(c, torch.tensor([0., .25]))
        self.assertGreaterEqual(float(b @ c), 0.)
        torch.testing.assert_close(m.compatible_response_gradient(b, e, 0.), torch.zeros(2))

    def test_zero_base_gradient_does_not_optimize_response_alone(self):
        c = m.compatible_response_gradient(torch.zeros(2), torch.ones(2), .25)
        torch.testing.assert_close(c, torch.zeros(2))

    def test_base_objective_contains_prediction_loss(self):
        envs = [dict(images=torch.zeros(12, 2, 14, 14),
                     labels=torch.tensor([[0.], [1.]]).repeat(6, 1)) for _ in range(2)]
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
        state = m.base_state(self.model, envs, 10000.)
        self.assertAlmostEqual(state["value"], np.log(2)/10000., places=10)
        self.assertGreater(state["value"], 0.)

    def test_zero_response_matches_original_training(self):
        import contextlib
        import io
        torch.manual_seed(3)
        envs = [dict(images=torch.rand(12, 2, 14, 14),
                     labels=(torch.rand(12, 1) > .5).float()) for _ in range(2)]
        args = SimpleNamespace(steps=3, warmup=1, gate_every=10, rho=1.,
                               response_weight=0., tau=1., log_every=10,
                               max_response_grad_ratio=.25)
        with contextlib.redirect_stdout(io.StringIO()):
            base = m.run("irm", self.model, envs, envs[0], envs, args, 3)
            disabled = m.run("blind", self.model, envs, envs[0], envs, args, 3)
        self.assertEqual(base["final_post_update"], disabled["final_post_update"])

    def test_all_methods_run_with_actual_response_and_guard(self):
        import contextlib
        import io
        torch.manual_seed(13)
        envs = [dict(images=torch.rand(12, 2, 14, 14),
                     labels=(torch.rand(12, 1) > .5).float()) for _ in range(2)]
        args = SimpleNamespace(steps=4, warmup=1, gate_every=1, rho=1.,
                               response_weight=1., tau=1., log_every=10,
                               max_response_grad_ratio=.25, backtracks=16,
                               armijo=1e-4, margin=0., fallback_step=.001,
                               max_step_norm=.1)
        with contextlib.redirect_stdout(io.StringIO()):
            for method in ("irm", "full_alignment", "full_compatible", "blind", "proxy_guard"):
                result = m.run(method, self.model, envs, envs[0], envs, args, 13)
                self.assertTrue(np.isfinite(result["final_post_update"]["target_accuracy"]))
                if method != "irm":
                    self.assertEqual(len(result["response_steps"]), 3)
                if method == "proxy_guard":
                    accepted = [r for r in result["guard_steps"] if r["applied"]]
                    self.assertGreater(len(accepted), 0)
                    for row in accepted:
                        self.assertLess(row["base_after"], row["base_before"])
                        self.assertLessEqual(row["base_after"], row["threshold"])

    def test_envelope_bounds_affine_targets_in_any_dimension(self):
        torch.manual_seed(19)
        for count in (2, 3, 7):
            risks = torch.rand(count, dtype=torch.float64)
            c = m.contrasts(count)
            radius = 3.
            bound = m.risk_envelope(risks, radius, 1e-4)
            for _ in range(10):
                a = torch.randn(count-1, dtype=torch.float64)
                a = radius * a / a.norm()
                weights = torch.ones(count, dtype=torch.float64)/count + c @ a
                self.assertLessEqual(float(weights @ risks), float(bound) + 1e-12)
            # The exact support function is attained along the risk contrast.
            delta = c.T @ risks
            exact = risks.mean() + radius * delta.norm()
            self.assertLessEqual(float(exact), float(bound))
            self.assertLessEqual(float(bound-exact), radius*1e-4)

    def test_new_guard_accepts_actual_irm_increase(self):
        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.5))

            def forward(self, images):
                return images * self.weight

        model = ScalarModel()
        envs = [dict(images=torch.ones(8, 1), labels=torch.ones(8, 1)) for _ in range(2)]
        args = SimpleNamespace(coverage=1., envelope_smoothing=1e-4, robust_l2=.001,
                               backtracks=16, armijo=1e-4, margin=0.,
                               fallback_step=.001, max_step_norm=.1)
        objective = lambda current: m.robust_state(current, envs, [], args, 1., 0.)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        old, state = m.vector(model), copy.deepcopy(optimizer.state_dict())
        irm_before = m.base_state(model, envs, 10000.)["value"]
        objective(model)["tensor"].backward()
        optimizer.step()
        row = m.guard_update(model, optimizer, old, state, envs, 0., args, state_fn=objective)
        self.assertTrue(row["applied"])
        self.assertLess(row["base_after"], row["base_before"])
        self.assertGreater(m.base_state(model, envs, 10000.)["value"], irm_before)

    def test_robust_methods_train_encoder_and_do_not_use_target_for_updates(self):
        import contextlib
        import io
        torch.manual_seed(29)
        envs = [dict(images=torch.rand(12, 2, 14, 14),
                     labels=(torch.rand(12, 1) > .5).float()) for _ in range(2)]
        args = SimpleNamespace(steps=3, rho=1., log_every=1, coverage=1.,
                               envelope_smoothing=1e-4, robust_l2=.001,
                               robust_response_weight=.01, backtracks=16,
                               armijo=1e-4, margin=0., fallback_step=.001, max_step_norm=.1)
        state = m.robust_state(self.model, envs, envs, args, 1., .01)
        grads = torch.autograd.grad(state["tensor"], tuple(self.model.parameters()))
        self.assertGreater(float(grads[0].norm()), 0.)
        self.assertGreater(float(grads[-1].norm()), 0.)
        alternate = dict(images=envs[0]["images"], labels=1-envs[0]["labels"])
        with contextlib.redirect_stdout(io.StringIO()):
            for method in m.ROBUST_METHODS:
                a = m.run(method, self.model, envs, envs[0], envs, args, 29)
                b = m.run(method, self.model, envs, alternate, envs, args, 29)
                self.assertEqual([r["objective_after"] for r in a["trace"]],
                                 [r["objective_after"] for r in b["trace"]])
                self.assertEqual(a["guard_steps"], b["guard_steps"])
                if method == "robust_guard":
                    accepted = [r for r in a["guard_steps"] if r["applied"]]
                    self.assertTrue(accepted)
                    for row in accepted:
                        self.assertLess(row["base_after"], row["base_before"])


if __name__ == "__main__":
    unittest.main()
