# Final results (A40, CARC)

Only the runs used in the final report and slides are committed here.
Earlier runs, taken before timing and memory were measured the same way
in every benchmark (see docs/measurement.md), are left out on purpose.

| Path | What it is |
|---|---|
| compare_deep10m/, compare_sift/ | Ours vs. Faiss SQ8/SQ4/Flat, cuVS Flat and cuVS RaBitQ at matched recall: QPS and peak GPU memory, batch 128 and all-at-once (b0 / b10000) |
| chain_*.csv | Kernel optimization chain: load width (v1 / v16) x top-C method (sort / select / fused) |
| vecexp_*.csv | Load width 1 / 4 / 16 bytes |
| topc_*.csv | Full sort vs. per-list radix select |
| rotexp_*.csv | 1-bit: rotation inside every scan block vs. query rotated once |
| rotation_ablation_*/ | 1-bit recall@10 with and without the random rotation |
| rabitq_warmup/ | cuVS RaBitQ with and without a warm-up index (speed is the same) |
| sweep_ours_*.csv | Our recall and QPS over nprobe and C, batch 32 / 128 / 512 |
| batch_*.csv | Our QPS across batch sizes |
| gpu_bruteforce_*.csv | GPU brute-force search used for ground truth |
| bandwidth_a40.csv | Measured device-memory bandwidth |
| ncu/*.csv | Nsight Compute kernel metrics |
