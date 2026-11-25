#!/usr/bin/env python3
"""Which configurations fit each GPU memory budget, and the fastest one per library (HS).

Reads the result CSVs of our scan (bench_scan: results/fix_*.csv), Faiss
(src/faiss_baseline.py) and cuVS (src/cuvs_baseline.py). A configuration fits budget B when
its device memory is at most B, where B is 1/16, 1/8, 1/4 or 1/2 of the raw FP32 dataset:

  ours   index_mb + work_mb       (bytes held by DeviceBudget)
  Faiss  gpu_peak_mb              (cudaMemGetInfo above the CUDA context, sampled)
  cuVS   gpu_peak_mb              (RMM statistics: index + queries + search allocations)
  CPU    0                        (Faiss CPU IVF uses no GPU memory)

For each budget and Recall@10 target, it prints the highest QPS that reaches the target,
with recall after the exact rerank. QPS covers the candidate search only (the rerank is not
timed) for every row used here; Faiss *_refine rows time the CPU rerank too and are left out.

  python scripts/budget_fit.py --dataset sift [--results results] [--csv results/budget_fit_sift.csv]
"""
import argparse
import glob
import os

import pandas as pd

RAW_MB = {"sift": 1_000_000 * 128 * 4 / 2**20, "deep10m": 10_000_000 * 96 * 4 / 2**20}
OURS = {"sift": "fix_sift_*.csv", "deep10m": "fix_deep10m_*.csv"}
FRACTIONS = [(1, 16), (1, 8), (1, 4), (1, 2)]
TARGETS = [0.90, 0.95, 0.99]


def load(results, ds):
    frames = []
    for path in sorted(glob.glob(os.path.join(results, OURS[ds]))):
        df = pd.read_csv(path)
        frames.append(pd.DataFrame({
            "library": "ours",
            "config": df["bits"].astype(str) + "-bit " + df["topc"],
            "nprobe": df["nprobe"], "C": df["C"], "qps": df["qps"],
            "recall": df["recall10_rerank"],
            "gpu_mb": df["index_mb"] + df["work_mb"],
            "source": os.path.basename(path)}))

    path = os.path.join(results, f"faiss_{ds}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["rerank_timed"] == 0]
        frames.append(pd.DataFrame({
            "library": "faiss", "config": df["algo"],
            "nprobe": df["nprobe"], "C": df["C"], "qps": df["qps"],
            "recall": df["recall10_rerank"],
            "gpu_mb": df["gpu_peak_mb"].fillna(0.0),
            "source": os.path.basename(path)}))

    path = os.path.join(results, f"cuvs_budget_{ds}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        frames.append(pd.DataFrame({
            "library": "cuvs", "config": df["algo"],
            "nprobe": df["nprobe"], "C": df["C"], "qps": df["qps"],
            "recall": df["recall10_rerank"], "gpu_mb": df["gpu_peak_mb"],
            "source": os.path.basename(path)}))
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(RAW_MB), required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--csv", default="")
    a = p.parse_args()

    rows = load(a.results, a.dataset)
    raw = RAW_MB[a.dataset]
    print(f"{a.dataset}: raw FP32 {raw:.1f} MB, {len(rows)} configurations")

    out = []
    for num, den in FRACTIONS:
        budget = raw * num / den
        fits = rows[rows["gpu_mb"] <= budget]
        print(f"\n=== budget {num}/{den} = {budget:.1f} MB: "
              f"{len(fits)} of {len(rows)} configurations fit")
        for target in TARGETS:
            ok = fits[fits["recall"] >= target]
            for lib in ["ours", "faiss", "cuvs"]:
                best = ok[ok["library"] == lib].sort_values("qps", ascending=False).head(1)
                if best.empty:
                    print(f"  R@10>={target:.2f}  {lib:<6} none")
                    out.append(dict(budget=f"{num}/{den}", budget_mb=round(budget, 1),
                                    target=target, library=lib))
                    continue
                r = best.iloc[0]
                print(f"  R@10>={target:.2f}  {lib:<6} {r.qps:>9.0f} qps  {r.config:<16} "
                      f"nprobe={r.nprobe:<3d} C={r.C:<4d} recall={r.recall:.4f} gpu={r.gpu_mb:.0f} MB")
                out.append(dict(budget=f"{num}/{den}", budget_mb=round(budget, 1), target=target,
                                library=lib, config=r.config, nprobe=r.nprobe, C=r.C,
                                qps=r.qps, recall=r.recall, gpu_mb=round(r.gpu_mb, 1),
                                source=r.source))
    if a.csv:
        pd.DataFrame(out).to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
