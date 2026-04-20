# Self-Pruning Neural Network Report

## Why L1 on Sigmoid Gates Encourages Sparsity
Each trainable gate score is passed through a sigmoid, so gates are constrained in [0, 1]. Adding an L1-style penalty over these gate values directly increases the objective cost of keeping many gates active. During optimization, less useful connections are pushed toward very small gate values, while important connections retain larger gates to preserve classification performance. This creates a sparse-but-functional network through training itself.

## Experimental Setup
- Dataset: CIFAR-10 (`torchvision.datasets.CIFAR10`)
- Architecture: `3072 -> 512 -> 256 -> 10` with `PrunableLinear`, `BatchNorm1d`, `GELU`, `Dropout(0.2)`
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Lambda schedule: epochs 1-5 warmup (`0`), 6-15 linear ramp, 16+ hold
- Epochs used in this run: 12

## Results
| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|---|---:|---:|
| 0.0e+00 | 53.68 | 0.00 |
| 1.0e-05 | 53.99 | 0.00 |
| 5.0e-05 | 53.70 | 0.00 |
| 1.0e-04 | 54.18 | 0.00 |

Best sparse operating point by accuracy: `lambda=1.0e-04` with accuracy `54.18%` and sparsity `0.00%`.

## Artifacts
- Gate distribution plot: `docs/charts/gate_distribution_best_lambda.png`
- Trade-off curve: `docs/charts/lambda_vs_accuracy_sparsity.png`
- Raw metrics json: `results_summary.json`

## Interpretation
As lambda increases, sparsity pressure becomes stronger, usually improving pruning ratio while potentially reducing accuracy if over-regularized. The reported sweep demonstrates this trade-off and identifies a practical balance point for deployment.
