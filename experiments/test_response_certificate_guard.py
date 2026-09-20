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
        self.args = SimpleNamespace(region_radius=100., backtracks=4, armijo=1e-4, margin=1e-7)

    def quadratic_proxy(self, model, *args, **kwargs):
        u = sum(p.double().square().sum() for p in model.parameters())/2
        value = float(u.detach())
        return {"tensor": u, "lower": value, "upper": value,
                "source_risks": [-value,-value], "D": value,
                "head_excess_lower": 0.,
                "optimum": {"head": m.head_vector(model), "gap":0., "converged":True}}

    def assert_optimizer_restored(self):
        actual = self.optimizer.state_dict()
        self.assertEqual(actual["param_groups"],self.state["param_groups"])
        for key, state in self.state["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value,actual["state"][key][name]))

    def test_rejection_restores_parameters_and_adam_state(self):
        self.args.margin = 1e9
        with patch.object(m,"proxy_state",side_effect=self.quadratic_proxy):
            row = m.guard_update(self.model,self.optimizer,self.old,self.state,[],torch.ones(1,1),self.args)
        self.assertEqual(row["status"],"NO_PROXY_DESCENT_FOUND")
        self.assertTrue(torch.equal(m.vector(self.model),self.old))
        self.assert_optimizer_restored()

    def test_fallback_accepts_proxy_descent_even_when_all_source_risks_rise(self):
        with patch.object(m,"proxy_state",side_effect=self.quadratic_proxy):
            row = m.guard_update(self.model,self.optimizer,self.old,self.state,[],torch.ones(1,1),self.args)
        self.assertEqual(row["status"],"FALLBACK_ACCEPTED")
        self.assertGreater(min(row["source_calibration_changes"]),0)
        self.assertLess(row["proxy_after_upper"],row["proxy_before_lower"]-row["armijo_decrease"])
        self.assert_optimizer_restored()

    def test_inner_uncertainty_can_reject_apparent_proxy_improvement(self):
        def uncertain(model,*args,**kwargs):
            row = self.quadratic_proxy(model)
            row["upper"] += 1e9
            row["optimum"]["gap"] = 1e9
            return row
        with patch.object(m,"proxy_state",side_effect=uncertain):
            row = m.guard_update(self.model,self.optimizer,self.old,self.state,[],torch.ones(1,1),self.args)
        self.assertFalse(row["applied"])
        self.assertTrue(any(c.get("proxy_lower",float("inf"))<row["proxy_before_lower"] for c in row["candidates"]))


if __name__ == "__main__":
    unittest.main()
