#!/usr/bin/env python3
"""Prints results 3, 4 and 5 (XJ) from the CSVs written by scripts/xj_pipeline_a40.sbatch.

  results/xj_rerank_breakdown.csv  -> result 3: CPU vs GPU rerank, split into parts
  results/xj_pipeline.csv          -> result 4: sequential vs pipelined, latency
  results/xj_rerank_scaling.csv    -> result 5: OpenMP scaling of the CPU rerank
  results/serial_<dataset>_a40.csv -> result 5: end-to-end speedup over the serial baseline

When a setting was run more than once, the last row wins.
Usage: python3 scripts/xj_summary.py [--results results]
"""
import argparse
import csv
import os


def read(path, key):
    """Rows of a CSV keyed by the given columns; later rows replace earlier ones."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {tuple(r[k] for k in key): r for r in csv.DictReader(f)}


def table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).rjust(w) for x, w in zip(r, widths))
    print(line(header))
    for r in rows:
        print(line(r))
    print()


def f(x, digits=2):
    return f"{float(x):.{digits}f}"


def bw(part_ms, total_ms, gbs):
    """Bandwidth from a time found by subtraction; too noisy when the part is under 20% of the total."""
    return f(gbs, 1) if float(part_ms) >= 0.2 * float(total_ms) else "-"


def result3(res):
    rows = read(os.path.join(res, "xj_rerank_breakdown.csv"), ("label", "bits", "C"))
    if not rows:
        return
    print("Result 3: rerank time (ms, all queries); CPU = ID download + gather + compute, "
          "GPU = PCIe + compute + result download")
    out = []
    for (label, bits, C), r in sorted(rows.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]))):
        cpu, gpu = float(r["cpu_path_ms"]), float(r["gpu_path_ms"])
        out.append([label, bits, C, f(r["recall10"], 4), f(r["cpu_id_d2h_ms"], 3), f(r["cpu_gather_ms"]),
                    f(r["cpu_compute_ms"]), f(cpu), f(r["gpu_pcie_ms"]), f(r["gpu_compute_ms"]),
                    f(r["gpu_result_d2h_ms"], 3), f(gpu), "CPU" if cpu < gpu else "GPU",
                    bw(r["cpu_gather_ms"], r["cpu_rerank_ms"], r["cpu_gather_gbs"]),
                    bw(r["gpu_pcie_ms"], r["gpu_kernel_ms"], r["gpu_pcie_gbs"]), r["rank_mismatch"]])
    table(["data", "bits", "C", "R@10", "cpu_ids", "gather", "compute", "CPU", "pcie", "compute", "results",
           "GPU", "faster", "gather_GB/s", "pcie_GB/s", "mismatch"], out)


def result4(res):
    rows = read(os.path.join(res, "xj_pipeline.csv"), ("label", "bits", "topc", "C", "batch", "mode"))
    if not rows:
        return
    print("Result 4: sequential vs pipelined (QPS; latency = per-batch scan start -> rerank end, ms)")
    out = []
    keys = sorted({k[:5] for k in rows}, key=lambda k: (k[0], int(k[1]), int(k[3]), int(k[4])))
    for k in keys:
        s, p = rows.get(k + ("sequential",)), rows.get(k + ("pipelined",))
        if not s or not p:
            continue
        out.append([k[0], k[1], k[2], k[3], k[4], f(s["recall10"], 4), f(s["qps"], 0), f(p["qps"], 0),
                    f(p["speedup"]), f(p["predicted_speedup"]), f(s["scan_ms"], 1), f(s["rerank_ms"], 1),
                    f(s["lat_p50_ms"], 3), f(s["lat_p99_ms"], 3), f(p["lat_p50_ms"], 3), f(p["lat_p99_ms"], 3),
                    p["mismatch"], p.get("seq_pipe_diff", "")])
    table(["data", "bits", "topc", "C", "batch", "R@10", "seq_qps", "pipe_qps", "speedup", "predicted",
           "scan_ms", "rerank_ms", "seq_p50", "seq_p99", "pipe_p50", "pipe_p99", "race", "tie_diff"], out)


def result_budget(res):
    rows = read(os.path.join(res, "xj_pipeline_budget.csv"), ("label", "bits", "topc", "C", "batch", "mode"))
    if not rows:
        return
    print("Pipeline under the GPU memory budget (label = dataset@fraction of the raw FP32 data)")
    out = []
    keys = sorted({k[:5] for k in rows}, key=lambda k: (k[0], int(k[1]), int(k[4])))
    for k in keys:
        over = rows.get(k + ("over_budget",)) or rows.get(k + ("unsupported",))
        s, p = rows.get(k + ("sequential",)), rows.get(k + ("pipelined",))
        if over and not s:
            status = "over" if over["mode"] == "over_budget" else "n/a"
            out.append([k[0], k[1], k[2], k[3], k[4], over["budget_mb"], status, "-", "-", "-", "-", "-", "-"])
        elif s and p:
            out.append([k[0], k[1], k[2], k[3], k[4], s["budget_mb"], "fits", f(s["index_mb"], 1),
                        f(s["work_mb"], 1), f(s.get("gpu_peak_mb") or s["measured_peak_mb"], 1),
                        f(s["recall10"], 4), f(s["qps"], 0), f(p["qps"], 0)])
    table(["data", "bits", "topc", "C", "batch", "budget_MiB", "status", "index_MiB", "work_MiB",
           "gpu_peak_MiB", "R@10", "seq_qps", "pipe_qps"], out)


def result5(res):
    rows = read(os.path.join(res, "xj_rerank_scaling.csv"), ("label", "C", "prefetch", "threads"))
    if rows:
        print("Result 5a: OpenMP scaling of the CPU rerank (1-bit candidates)")
        out = [[k[0], k[1], k[2], k[3], f(r["ms"]), f(r["speedup"]), f(r["efficiency"]), f(r["gbs"], 1)]
               for k, r in sorted(rows.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]),
                                                                  int(kv[0][3])))]
        table(["data", "C", "prefetch", "threads", "ms", "speedup", "efficiency", "GB/s"], out)

    pipe = read(os.path.join(res, "xj_pipeline.csv"), ("label", "bits", "topc", "C", "batch", "mode"))
    out = []
    for label in sorted({k[0] for k in pipe}):
        serial = read(os.path.join(res, f"serial_{label}_a40.csv"), ("n",))
        if not serial:
            continue
        sqps = float(list(serial.values())[-1]["qps"])
        for (lab, bits, topc, C, batch, mode), r in sorted(pipe.items()):
            if lab != label or mode != "pipelined":
                continue
            out.append([label, bits, topc, C, batch, f(r["recall10"], 4), f(sqps, 1), f(r["qps"], 0),
                        f(float(r["qps"]) / sqps, 0)])
    if out:
        print("Result 5b: end-to-end speedup of the pipelined search over the serial exact baseline")
        table(["data", "bits", "topc", "C", "batch", "R@10", "serial_qps", "ours_qps", "speedup"], out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = ap.parse_args().results
    result3(res)
    result4(res)
    result_budget(res)
    result5(res)


if __name__ == "__main__":
    main()
