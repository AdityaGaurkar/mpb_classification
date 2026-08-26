"""Measure deployment-relevant metrics (params, FLOPs, latency, model size)
for the FEv4 custom CNN and the lightweight baselines.

Outputs: outputs/deployment_metrics.csv
Usage:   python revision/deploy_metrics.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_SIZE, ROOT, make_model

RESULTS = os.path.join(ROOT, "outputs", "deployment_metrics.csv")


def flops_forward(model, x_t):
    """Multiply-accumulate FLOPs (2*MACs) summed over Conv2d/Linear layers."""
    total = 0

    def hook(module, inp, out):
        nonlocal total
        if isinstance(module, torch.nn.Conv2d):
            ow, oh = out.shape[-2:]
            total += 2 * module.kernel_size[0] * module.kernel_size[1] \
                * module.in_channels * module.out_channels * ow * oh
        elif isinstance(module, torch.nn.Linear):
            total += 2 * module.in_features * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        with torch.no_grad():
            model(x_t)
    finally:
        for h in handles:
            h.remove()
    return total


def time_inference(model, x, device, iters=50):
    m = model.to(device)
    xx = x.to(device)
    for _ in range(3):
        m(xx)
    t0 = time.perf_counter()
    for _ in range(iters):
        m(xx)
    dt = (time.perf_counter() - t0) / iters * 1000
    m.to("cpu")
    return dt


def time_infer_mps(model, channels):
    return time_inference(model, torch.randn(1, channels, IMG_SIZE, IMG_SIZE),
                          torch.device("mps"))


def main():
    torch.manual_seed(0)
    rows = []
    has_mps = torch.backends.mps.is_available()
    for name in ("fe4", "squeezenet1_1", "mobilenet_v3_small", "efficientnet_b0"):
        model, channels, _ = make_model(name)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        flops = flops_forward(model, torch.randn(1, channels, IMG_SIZE, IMG_SIZE))
        size_mb = params * 4 / 1e6  # fp32
        mps = time_infer_mps(model, channels) if has_mps else float("nan")
        cpu = time_inference(model, torch.randn(1, channels, IMG_SIZE, IMG_SIZE),
                             torch.device("cpu"))
        rows.append({
            "model": name, "params_M": params / 1e6, "flops_M": flops / 1e6,
            "size_MB_fp32": size_mb, "latency_mps_ms": mps, "latency_cpu_ms": cpu,
        })
        print(f"{name:<20} params={params/1e6:6.2f}M flops={flops/1e6:9.1f}M "
              f"size={size_mb:6.1f}MB mps={mps:7.2f}ms cpu={cpu:7.2f}ms")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    import csv

    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {RESULTS}")


if __name__ == "__main__":
    main()