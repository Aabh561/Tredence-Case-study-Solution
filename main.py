from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel() * t.element_size())


class PrunableLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        temperature: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.temperature = float(temperature)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, 5.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self) -> Tensor:
        return torch.sigmoid(self.gate_scores / self.temperature)

    def forward(self, x: Tensor) -> Tensor:
        gates = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def compress_to_sparse(self, threshold: float = 1e-2) -> torch.Tensor:
        with torch.no_grad():
            gates = self.get_gates()
            mask = gates >= threshold
            sparse_dense = (self.weight * mask).to(dtype=self.weight.dtype)
            return sparse_dense.to_sparse_csr()

    def dense_memory_bytes(self) -> int:
        total = tensor_nbytes(self.weight)
        total += tensor_nbytes(self.gate_scores)
        if self.bias is not None:
            total += tensor_nbytes(self.bias)
        return total

    @staticmethod
    def csr_memory_bytes(csr_tensor: torch.Tensor) -> int:
        if csr_tensor.layout != torch.sparse_csr:
            raise ValueError("csr_tensor must have sparse_csr layout.")
        return (
            tensor_nbytes(csr_tensor.values())
            + tensor_nbytes(csr_tensor.col_indices())
            + tensor_nbytes(csr_tensor.crow_indices())
        )

    def memory_report_bytes(self, threshold: float = 1e-2) -> dict[str, int]:
        csr = self.compress_to_sparse(threshold=threshold)
        dense_bytes = self.dense_memory_bytes()
        compressed_bytes = self.csr_memory_bytes(csr)
        return {
            "dense_bytes": dense_bytes,
            "csr_bytes": compressed_bytes,
            "bytes_saved": dense_bytes - compressed_bytes,
        }


class SelfPruningMLP(nn.Module):
    def __init__(self, temperature: float = 1.0, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512, temperature=temperature)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256, temperature=temperature)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 10, temperature=temperature)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def get_sparsity_loss(self) -> Tensor:
        loss_terms: list[Tensor] = []
        for name, param in self.named_parameters():
            if "gate_scores" in name:
                loss_terms.append(torch.sigmoid(param).abs().sum())
        if not loss_terms:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(loss_terms).sum()

    def count_gate_params(self) -> int:
        total = 0
        for name, param in self.named_parameters():
            if "gate_scores" in name:
                total += int(param.numel())
        return total

    def collect_gate_values(self) -> Tensor:
        gates: list[Tensor] = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().detach().flatten())
        return torch.cat(gates) if gates else torch.empty(0)

    def memory_footprint(self, threshold: float = 1e-2) -> dict[str, int]:
        dense_total = 0
        csr_total = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                report = module.memory_report_bytes(threshold=threshold)
                dense_total += report["dense_bytes"]
                csr_total += report["csr_bytes"]
        return {
            "dense_bytes": dense_total,
            "csr_bytes": csr_total,
            "bytes_saved": dense_total - csr_total,
        }


def lambda_schedule(epoch: int, target_lambda: float) -> float:
    if epoch <= 5:
        return 0.0
    if epoch <= 15:
        alpha = (epoch - 5) / 10.0
        return float(alpha * target_lambda)
    return float(target_lambda)


@dataclass
class ExperimentConfig:
    batch_size: int = 256
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    gate_threshold: float = 1e-2
    chart_dir: Path = Path("docs/charts")
    report_path: Path = Path("self_pruning_report.md")
    results_json_path: Path = Path("results_summary.json")
    clip_gate_grad_norm: float = 1.0


def _seed_worker(worker_id: int) -> None:
    # Keep dataloader randomness deterministic per worker.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: SelfPruningMLP,
    loader: DataLoader[Any],
    optimizer: AdamW,
    device: torch.device,
    target_lambda: float,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    lambda_t = lambda_schedule(epoch, target_lambda)
    gate_param_count = max(model.count_gate_params(), 1)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        sparse_loss = model.get_sparsity_loss() / float(gate_param_count)
        loss = ce_loss + (lambda_t * sparse_loss)
        loss.backward()
        gate_params = [
            p for name, p in model.named_parameters() if "gate_scores" in name and p.grad is not None
        ]
        if gate_params:
            torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: SelfPruningMLP,
    loader: DataLoader[Any],
    device: torch.device,
    gate_threshold: float = 1e-2,
) -> dict[str, float | dict[str, int] | Tensor]:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    gate_values = model.collect_gate_values().cpu()
    sparsity = float((gate_values < gate_threshold).float().mean().item() * 100.0)
    mean_gate = float(gate_values.mean().item()) if gate_values.numel() else 0.0
    memory = model.memory_footprint(threshold=gate_threshold)
    accuracy = 100.0 * correct / max(total, 1)
    return {
        "test_accuracy": float(accuracy),
        "sparsity_percent": sparsity,
        "mean_gate_value": mean_gate,
        "memory": memory,
        "gate_values": gate_values,
    }


def run_experiment(
    target_lambda: float,
    cfg: ExperimentConfig,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(cfg.device)
    model = SelfPruningMLP(temperature=cfg.temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            target_lambda=target_lambda,
            epoch=epoch,
        )
        scheduler.step()
        if epoch in {1, cfg.epochs // 2, cfg.epochs}:
            print(
                f"[lambda={target_lambda}] epoch={epoch}/{cfg.epochs} "
                f"train_loss={epoch_loss:.4f} lambda_t={lambda_schedule(epoch, target_lambda):.2e}"
                ,
                flush=True,
            )

    eval_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        gate_threshold=cfg.gate_threshold,
    )
    metrics = {
        "lambda": target_lambda,
        "test_accuracy": float(eval_metrics["test_accuracy"]),
        "sparsity_percent": float(eval_metrics["sparsity_percent"]),
        "mean_gate_value": float(eval_metrics["mean_gate_value"]),
        "memory": eval_metrics["memory"],
    }
    artifacts = {
        "model_state": copy.deepcopy(model.state_dict()),
        "gate_values": eval_metrics["gate_values"],
    }
    return metrics, artifacts


def run_lambda_sweep(
    lambdas: list[float],
    cfg: ExperimentConfig,
) -> dict[float, dict[str, Any]]:
    set_determinism(cfg.seed)
    cfg.chart_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = build_dataloaders(cfg)

    results: dict[float, dict[str, Any]] = {}
    for lam in lambdas:
        metrics, artifacts = run_experiment(lam, cfg, train_loader, test_loader)
        results[lam] = {**metrics, **artifacts}

    best_lambda, best_data = max(results.items(), key=lambda item: item[1]["test_accuracy"])
    plot_gate_distribution(
        gate_values=best_data["gate_values"],
        out_path=cfg.chart_dir / "gate_distribution_best_lambda.png",
        title=f"Gate Distribution (lambda={best_lambda})",
    )

    plot_tradeoff_curve(
        results=results,
        out_path=cfg.chart_dir / "lambda_vs_accuracy_sparsity.png",
    )
    summary = strip_heavy_artifacts(results)
    save_results_json(summary, cfg.results_json_path)
    write_submission_report(summary, cfg)
    return results


def plot_gate_distribution(gate_values: Tensor, out_path: Path, title: str) -> None:
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = gate_values.numpy()
    ax.hist(vals, bins=200, range=(0.0, 1.0), color="#1f77b4", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Gate value (sigmoid(gate_scores))")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tradeoff_curve(results: dict[float, dict[str, Any]], out_path: Path) -> None:
    sns.set_theme(style="darkgrid")
    lambdas = sorted(results.keys())
    accuracies = [results[l]["test_accuracy"] for l in lambdas]
    sparsities = [results[l]["sparsity_percent"] for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(lambdas, accuracies, marker="o", linewidth=2.2, color="#2ca02c")
    ax2.plot(lambdas, sparsities, marker="s", linewidth=2.2, color="#d62728")

    ax1.set_xlabel("Target lambda")
    ax1.set_ylabel("Test Accuracy (%)")
    ax2.set_ylabel("Sparsity Level (%)")
    ax1.set_title("Lambda vs Accuracy and Sparsity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def strip_heavy_artifacts(results: dict[float, dict[str, Any]]) -> dict[float, dict[str, Any]]:
    cleaned: dict[float, dict[str, Any]] = {}
    for lam, data in results.items():
        cleaned[lam] = {
            "lambda": data["lambda"],
            "test_accuracy": data["test_accuracy"],
            "sparsity_percent": data["sparsity_percent"],
            "mean_gate_value": data["mean_gate_value"],
            "memory": data["memory"],
        }
    return cleaned


def save_results_json(results: dict[float, dict[str, Any]], out_path: Path) -> None:
    serializable = {str(k): v for k, v in results.items()}
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def write_submission_report(results: dict[float, dict[str, Any]], cfg: ExperimentConfig) -> None:
    sorted_lambdas = sorted(results.keys())
    table_rows = "\n".join(
        [
            f"| {lam:.1e} | {results[lam]['test_accuracy']:.2f} | {results[lam]['sparsity_percent']:.2f} |"
            for lam in sorted_lambdas
        ]
    )
    best_sparse = [
        (lam, metrics)
        for lam, metrics in results.items()
        if metrics["sparsity_percent"] > 0.0
    ]
    if best_sparse:
        best_lambda, best_metrics = max(best_sparse, key=lambda item: item[1]["test_accuracy"])
    else:
        best_lambda, best_metrics = max(results.items(), key=lambda item: item[1]["test_accuracy"])

    report = f"""# Self-Pruning Neural Network Report

## Why L1 on Sigmoid Gates Encourages Sparsity
Each trainable gate score is passed through a sigmoid, so gates are constrained in [0, 1]. Adding an L1-style penalty over these gate values directly increases the objective cost of keeping many gates active. During optimization, less useful connections are pushed toward very small gate values, while important connections retain larger gates to preserve classification performance. This creates a sparse-but-functional network through training itself.

## Experimental Setup
- Dataset: CIFAR-10 (`torchvision.datasets.CIFAR10`)
- Architecture: `3072 -> 512 -> 256 -> 10` with `PrunableLinear`, `BatchNorm1d`, `GELU`, `Dropout(0.2)`
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Lambda schedule: epochs 1-5 warmup (`0`), 6-15 linear ramp, 16+ hold
- Epochs used in this run: {cfg.epochs}

## Results
| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---|---:|---:|
{table_rows}

Best sparse operating point by accuracy: `lambda={best_lambda:.1e}` with accuracy `{best_metrics['test_accuracy']:.2f}%` and sparsity `{best_metrics['sparsity_percent']:.2f}%`.

## Artifacts
- Gate distribution plot: `docs/charts/gate_distribution_best_lambda.png`
- Trade-off curve: `docs/charts/lambda_vs_accuracy_sparsity.png`
- Raw metrics json: `{cfg.results_json_path.as_posix()}`

## Interpretation
As lambda increases, sparsity pressure becomes stronger, usually improving pruning ratio while potentially reducing accuracy if over-regularized. The reported sweep demonstrates this trade-off and identifies a practical balance point for deployment.
"""
    cfg.report_path.write_text(report, encoding="utf-8")


def main() -> None:
    cfg = ExperimentConfig()
    target_lambdas = [0.0, 1e-5, 5e-5, 1e-4]
    raw_results = run_lambda_sweep(target_lambdas, cfg)
    compact_results = strip_heavy_artifacts(raw_results)
    print("\n=== Final Sweep Results ===")
    for lam, metrics in compact_results.items():
        print(f"lambda={lam}: {metrics}")
    print(f"\nSaved report to: {cfg.report_path}")
    print(f"Saved metrics to: {cfg.results_json_path}")


if __name__ == "__main__":
    main()
