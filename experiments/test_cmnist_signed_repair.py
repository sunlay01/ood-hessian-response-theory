import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import run_cmnist_signed_repair as m


class CMNISTRepairTests(unittest.TestCase):
    def test_official_validation_is_training_tail(self):
        a, b, val = m.official_indices(0)
        self.assertEqual(len(a), 25000)
        self.assertEqual(len(b), 25000)
        self.assertEqual(len(set(a.tolist()) & set(b.tolist())), 0)
        self.assertEqual(set(a.tolist()+b.tolist()), set(range(50000)))
        self.assertTrue(torch.equal(val, torch.arange(50000, 60000)))

    def test_full_encoder_offset_is_in_certificate(self):
        # Pre-update risk=.6; mixed risk=log(2), so a head correction must
        # first compensate the encoder's +.093 risk increase.
        xs = [np.ones((20, 1))] * 2
        ys = [np.ones(20)] * 2
        result = m.solve_head_repair(xs, ys, [.6, .6],
                                     np.zeros(1), np.array([-1.]), .001)
        self.assertEqual(result["status"], "REPAIRED")
        self.assertIsNotNone(result["head"])
        self.assertGreater(result["encoder_risk_offsets"][0], .09)
        self.assertLess(m.logistic_risk(xs[0], ys[0], result["head"]), .599)

    def test_no_repair_at_irreducible_balanced_intercept(self):
        xs = [np.ones((20, 1))] * 2
        ys = [np.tile([0., 1.], 10)] * 2
        result = m.solve_head_repair(xs, ys, [np.log(2)]*2,
                                     np.zeros(1), np.zeros(1), 1e-6)
        self.assertIsNone(result["head"])

    def test_disabled_guard_identical_to_irm(self):
        torch.manual_seed(7)
        source = [{"images": torch.rand(12, 2, 14, 14),
                   "labels": (torch.rand(12, 1) > .5).float()} for _ in range(2)]
        initial = m.MLP(4)
        args = SimpleNamespace(steps=2, repair_start=10, rho=1e-6, log_every=10)
        with contextlib.redirect_stdout(io.StringIO()):
            base = m.run_method("irm", initial, source, source[0], args, 7)
            disabled = m.run_method("irm_signed", initial, source, source[0], args, 7)
        self.assertEqual(base["final_post_update"], disabled["final_post_update"])
        self.assertEqual(base["official_final_pre_update"], disabled["official_final_pre_update"])
        self.assertEqual(disabled["guard_steps"], [])

    def test_rejected_proposal_restores_encoder_and_head(self):
        torch.manual_seed(19)
        model = m.MLP(4)
        source = [{"images": torch.rand(12, 2, 14, 14),
                   "labels": torch.ones(12, 1)} for _ in range(2)]
        snapshot = [p.detach().clone() for p in model.parameters()]
        old_head = m.head_vector(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        m.mean_nll(model(source[0]["images"]), source[0]["labels"]).backward()
        optimizer.step()
        moments = [s["exp_avg"].clone() for s in optimizer.state.values()]
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(model.parameters(), snapshot)))
        with patch.object(m, "solve_head_repair", return_value={
            "status": "NO_CERTIFIED_REPAIR", "head": None, "attempts": []
        }):
            result = m.apply_guard(model, snapshot, old_head, [.7, .7], source, 1e-6)
        self.assertFalse(result["applied"])
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(model.parameters(), snapshot)))
        self.assertTrue(all(torch.equal(s["exp_avg"], q) for s, q in zip(optimizer.state.values(), moments)))
        self.assertTrue(all(s["step"].item() == 1 for s in optimizer.state.values()))

    def test_active_repair_changes_proposal_and_lowers_full_risk(self):
        torch.manual_seed(23)
        model = m.MLP(4)
        source = [{"images": torch.rand(12, 2, 14, 14),
                   "labels": torch.ones(12, 1)} for _ in range(2)]
        m.set_head(model, np.zeros(5))
        old_head = m.head_vector(model)
        snapshot = [p.detach().clone() for p in model.parameters()]
        # Perturb both encoder layers and propose a deliberately harmful head.
        with torch.no_grad():
            for p in list(model.parameters())[:4]:
                p.add_(.001)
        m.set_head(model, np.full(5, -.05))
        result = m.apply_guard(model, snapshot, old_head, [np.log(2)]*2, source, 1e-4)
        self.assertEqual(result["status"], "REPAIRED")
        self.assertTrue(result["applied"])
        self.assertGreater(result["head_correction_norm"], 0)
        self.assertFalse(result["bound_violation"])
        with torch.no_grad():
            for env in source:
                self.assertLess(float(m.mean_nll(model(env["images"]), env["labels"])), np.log(2)-1e-4)


if __name__ == "__main__":
    unittest.main()
