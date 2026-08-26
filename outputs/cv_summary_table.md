# 5-fold identity-grouped cross-validation results

Every identity is predicted exactly once across folds (pooled columns); 
± values are mean ± std across the 5 folds.

| Model | Acc (pooled) | 95% CI | Acc (CV) | Macro-F1 | F1 (CV) | Within-1 stage | Within-1 (CV) | MAE | MAE (CV) |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN (FEv4, from scratch) | 71.54% | [63.3, 78.6] | 71.48 ± 5.18% | 0.712 | 0.703 ± 0.058 | 96.92% | 96.93 ± 2.74% | 0.315 | 0.316 ± 0.079 |
| SqueezeNet 1.1 (ImageNet) | 74.62% | [66.5, 81.3] | 75.30 ± 9.52% | 0.753 | 0.750 ± 0.091 | 98.46% | 98.57 ± 2.86% | 0.269 | 0.261 ± 0.110 |
| MobileNetV3-Small (ImageNet) | 69.23% | [60.8, 76.5] | 69.10 ± 5.06% | 0.630 | 0.611 ± 0.090 | 96.92% | 96.84 ± 1.61% | 0.346 | 0.349 ± 0.065 |
| EfficientNet-B0 (ImageNet) | 76.15% | [68.1, 82.7] | 76.26 ± 3.14% | 0.758 | 0.753 ± 0.036 | 100.00% | 100.00 ± 0.00% | 0.238 | 0.237 ± 0.031 |

**Custom CNN (FEv4, from scratch)** — error structure: 37 errors total; |Δ|=1: 89.2%; |Δ|=2: 10.8%; |Δ|=3: 0.0%
**SqueezeNet 1.1 (ImageNet)** — error structure: 33 errors total; |Δ|=1: 93.9%; |Δ|=2: 6.1%; |Δ|=3: 0.0%
**MobileNetV3-Small (ImageNet)** — error structure: 40 errors total; |Δ|=1: 90.0%; |Δ|=2: 7.5%; |Δ|=3: 2.5%
**EfficientNet-B0 (ImageNet)** — error structure: 31 errors total; |Δ|=1: 100.0%; |Δ|=2: 0.0%; |Δ|=3: 0.0%
