"""Canonical Colored-MNIST IRMv1 calibration.

This runner is intentionally separate from the response-method comparison.  It
implements the reference IRMv1 protocol closely enough to diagnose baseline
fidelity before comparing any new method:

* 50,000 shuffled MNIST training examples split into two 25,000-example
  environments with color flip probabilities .2 and .1;
* 25% label noise in every environment and .9 color flip on test;
* 256-256-1 MLP with Xavier initialization;
* Adam, lr=.001, 501 steps, L2=.001, penalty annealing at step 100, and
  penalty weight 10,000 with the reference loss rescaling.

The output contains raw per-seed traces so a low result can be audited rather
than hidden behind a summary table.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from torch import autograd, nn
from torch.nn import functional as F


torch.set_num_threads(4)


def read_idx(path: Path, image: bool) -> torch.Tensor:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        if image:
            magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
            if magic != 2051:
                raise ValueError(f"invalid image magic: {magic}")
            values = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, rows, cols)
        else:
            magic, count = struct.unpack(">II", handle.read(8))
            if magic != 2049:
                raise ValueError(f"invalid label magic: {magic}")
            values = np.frombuffer(handle.read(), dtype=np.uint8).reshape(count)
    return torch.from_numpy(values.copy())


def load_mnist(root: Path):
    raw = root / "MNIST" / "raw"
    return (
        read_idx(raw / "train-images-idx3-ubyte", True),
        read_idx(raw / "train-labels-idx1-ubyte", False).long(),
        read_idx(raw / "t10k-images-idx3-ubyte", True),
        read_idx(raw / "t10k-labels-idx1-ubyte", False).long(),
    )


def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs()


def make_environment(
    images: torch.Tensor,
    digits: torch.Tensor,
    indices: torch.Tensor,
    color_flip: float,
    generator: torch.Generator,
    label_noise: float = 0.25,
) -> dict[str, torch.Tensor]:
    gray = images[indices, ::2, ::2].float() / 255.0
    labels = (digits[indices] < 5).float()
    labels = xor(labels, (torch.rand(len(indices), generator=generator) < label_noise).float())
    colors = xor(labels, (torch.rand(len(indices), generator=generator) < color_flip).float())
    stacked = torch.stack([gray, gray], dim=1)
    stacked[torch.arange(len(indices)), (1 - colors).long()] = 0.0
    return {"images": stacked, "labels": labels[:, None]}


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        layers = [
            nn.Linear(2 * 14 * 14, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        ]
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.main = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.main(images.flatten(1))


def mean_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def mean_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(((logits > 0).float() == labels).float().mean())


def penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scale = torch.ones((), requires_grad=True)
    value = mean_nll(logits * scale, labels)
    grad = autograd.grad(value, [scale], create_graph=True)[0]
    return grad.pow(2).sum()


def build_envs(
    data_root: Path, seed: int, label_noise: float, n_source: int, n_target: int
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    train_images, train_digits, test_images, test_digits = load_mnist(data_root)
    official_train_count = min(50000, len(train_digits))
    if n_source * 2 > official_train_count:
        raise ValueError("n_source * 2 exceeds the official 50,000-example train split")
    split_generator = torch.Generator().manual_seed(seed + 1000)
    shuffled = torch.randperm(official_train_count, generator=split_generator)[: 2 * n_source]
    env_indices = [shuffled[:n_source], shuffled[n_source:]]
    envs = [
        make_environment(
            train_images,
            train_digits,
            indices,
            flip,
            torch.Generator().manual_seed(seed + 2000 + i),
            label_noise,
        )
        for i, (indices, flip) in enumerate(zip(env_indices, (0.2, 0.1)))
    ]
    test_generator = torch.Generator().manual_seed(seed + 3000)
    if n_target < len(test_digits):
        target_indices = torch.randperm(len(test_digits), generator=test_generator)[:n_target]
    else:
        target_indices = torch.arange(len(test_digits))
    target = make_environment(
        test_images,
        test_digits,
        target_indices,
        0.9,
        torch.Generator().manual_seed(seed + 4000),
        label_noise,
    )
    return envs, target


def evaluate(model: nn.Module, envs, target) -> dict[str, float]:
    with torch.no_grad():
        source_nll = [float(mean_nll(model(env["images"]), env["labels"])) for env in envs]
        target_logits = model(target["images"])
        target_nll = float(mean_nll(target_logits, target["labels"]))
        source_accuracy = [mean_accuracy(model(env["images"]), env["labels"]) for env in envs]
        target_accuracy = mean_accuracy(target_logits, target["labels"])
    return {
        "source_nll": float(np.mean(source_nll)),
        "target_nll": target_nll,
        "source_accuracy": float(np.mean(source_accuracy)),
        "target_accuracy": target_accuracy,
    }


def run(seed: int, args) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs, target = build_envs(
        args.data_root, seed, args.label_noise, args.n_source, args.n_target
    )
    model = MLP(args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trace = []
    started = time.time()
    for step in range(args.steps):
        logits = [model(env["images"]) for env in envs]
        nlls = [mean_nll(value, env["labels"]) for value, env in zip(logits, envs)]
        penalties = [penalty(value, env["labels"]) for value, env in zip(logits, envs)]
        train_nll = torch.stack(nlls).mean()
        train_penalty = torch.stack(penalties).mean()
        weight_norm = sum(parameter.norm().pow(2) for parameter in model.parameters())
        penalty_weight = args.penalty_weight if step >= args.penalty_anneal_iters else 1.0
        objective = train_nll + args.l2_weight * weight_norm + penalty_weight * train_penalty
        if penalty_weight > 1.0:
            objective = objective / penalty_weight
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            current = evaluate(model, envs, target)
            current.update(
                {
                    "step": step,
                    "train_nll": float(train_nll.detach()),
                    "train_penalty": float(train_penalty.detach()),
                    "penalty_weight": float(penalty_weight),
                    "objective": float(objective.detach()),
                }
            )
            trace.append(current)
            print(
                f"seed={seed} step={step:04d} nll={current['train_nll']:.4f} "
                f"penalty={current['train_penalty']:.6g} "
                f"target_acc={current['target_accuracy']:.4f}",
                flush=True,
            )
    result = {
        "seed": seed,
        "final": evaluate(model, envs, target),
        "trace": trace,
        "wall_seconds": time.time() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/sunlay/Desktop/data"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/cmnist_irm_calibration"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-source", type=int, default=25000)
    parser.add_argument("--n-target", type=int, default=10000)
    parser.add_argument("--label-noise", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--penalty-anneal-iters", type=int, default=100)
    parser.add_argument("--penalty-weight", type=float, default=10000.0)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run(seed, args) for seed in args.seeds]
    payload = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "results": results,
    }
    (args.output / "all_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = [result["final"] | {"seed": result["seed"], "wall_seconds": result["wall_seconds"]} for result in results]
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"configuration": payload["configuration"], "final": rows}, indent=2))


if __name__ == "__main__":
    main()
