#!/usr/bin/env python3
"""Summarize an Nsight Compute CSV (ncu --csv --page raw) per kernel.

For each kernel: launches, total GPU time, DRAM bytes read and written, the achieved
DRAM bandwidth, and that bandwidth as a share of the A40's 696 GB/s peak.

Usage: ncu_summary.py profile.csv [--peak-gbs 696] [--out summary.csv]
"""
import argparse
import csv
import re
from collections import OrderedDict

TIME_TO_MS = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0,
              "s": 1e3, "second": 1e3}
TO_BYTES = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


def number(text):
    text = text.replace(",", "").strip()
    return float(text) if text not in ("", "n/a") else 0.0


def short_name(full):
    """'void <unnamed>::scan_kernel<8>(<unnamed>::ScanParams)' -> 'scan_kernel<8>';
    'void cub::...::DeviceSegmentedRadixSortKernel<...>(...)' -> 'DeviceSegmentedRadixSortKernel'."""
    name = re.sub(r"^void ", "", full)
    name = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    name = name.split("(")[0]               # drop the argument list
    base = name.split("<")[0].split("::")[-1]
    bits = re.match(r"scan_kernel<(\d+)>", name)
    return f"{base}<{bits.group(1)}>" if bits else base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--peak-gbs", type=float, default=696.0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    with open(a.csv, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    start = next(i for i, r in enumerate(rows) if "Kernel Name" in r)  # ncu may print log lines first
    header, units, data = rows[start], rows[start + 1], rows[start + 2:]
    col = {name: i for i, name in enumerate(header)}

    def value(row, metric, scale):
        i = col[metric]
        return number(row[i]) * scale.get(units[i], 1.0)

    kernels = OrderedDict()
    for r in data:
        k = kernels.setdefault(short_name(r[col["Kernel Name"]]),
                               {"n": 0, "ms": 0.0, "rd": 0.0, "wr": 0.0, "pct": 0.0})
        k["n"] += 1
        k["ms"] += value(r, "gpu__time_duration.sum", TIME_TO_MS)
        k["rd"] += value(r, "dram__bytes_read.sum", TO_BYTES)
        k["wr"] += value(r, "dram__bytes_write.sum", TO_BYTES)
        k["pct"] += number(r[col["dram__throughput.avg.pct_of_peak_sustained_elapsed"]])

    total_ms = sum(k["ms"] for k in kernels.values())
    table = []
    print(f"{'kernel':34s} {'launches':>8s} {'time ms':>9s} {'share':>6s} {'read MB':>9s} "
          f"{'write MB':>9s} {'GB/s':>7s} {'of peak':>8s} {'ncu dram%':>9s}")
    for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]["ms"]):
        gbs = (k["rd"] + k["wr"]) / 1e9 / (k["ms"] / 1e3) if k["ms"] > 0 else 0.0
        row = [name, k["n"], k["ms"], 100 * k["ms"] / total_ms, k["rd"] / 1e6, k["wr"] / 1e6,
               gbs, 100 * gbs / a.peak_gbs, k["pct"] / k["n"]]
        table.append(row)
        print(f"{name[:34]:34s} {row[1]:8d} {row[2]:9.3f} {row[3]:5.1f}% {row[4]:9.1f} "
              f"{row[5]:9.1f} {row[6]:7.1f} {row[7]:7.1f}% {row[8]:8.1f}%")
    print(f"total GPU time {total_ms:.3f} ms over {sum(k['n'] for k in kernels.values())} launches")

    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "launches", "time_ms", "time_share_pct", "read_mb", "write_mb",
                        "dram_gbs", "pct_of_peak", "ncu_dram_pct"])
            w.writerows(table)


if __name__ == "__main__":
    main()
