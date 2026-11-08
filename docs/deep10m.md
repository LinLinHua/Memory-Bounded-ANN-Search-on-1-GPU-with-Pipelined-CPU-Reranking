# DEEP10M data preparation

DEEP10M is the first 10,000,000 database vectors from the DEEP1B fbin file,
in original order, paired with the public 10,000-query set. Keep the original
float32 values; do not normalize, shuffle, or change vector IDs.

Source: <https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/>

| Output under `data/deep10m/` | Shape | Exact size |
| --- | --- | --- |
| `base.10M.fbin` | 10,000,000 x 96 | 3,840,000,008 bytes |
| `query.public.10K.fbin` | 10,000 x 96 | 3,840,008 bytes |

Each file begins with two little-endian int32 values (rows, dimensions),
followed by row-major float32 data. The cropped base header is changed from
1,000,000,000 rows to 10,000,000. IDs are zero-based positions in this prefix.

## Prepare on CARC

From the project root:

```bash
mkdir -p logs
sbatch scripts/prepare_deep10m.sbatch
```

The single-core CPU job uses four concurrent HTTP Range requests and never
downloads the full 384 GB base. Every response must have the requested byte
range and expected source size. Incomplete files use `.part`; rerunning resumes
them. Do not run multiple preparation jobs against the same output directory.

After download, the script validates file sizes, headers, and every float for
NaN/infinity, and writes `manifest.json` with the source URLs, selection rule,
and SHA-256 digests. These are locally computed digests, not independently
published source checksums. An existing final file is validated rather than
overwritten. `manifest.json` is written only after both files validate.

For the shared checkout at `/project2/prasanna_933/ANN-ac`, copies of the scripts
are also stored in the ignored data directory, avoiding changes to teammates'
tracked files. Submit those copies with:

```bash
DEEP_PREPARE_SCRIPT=data/deep10m/prepare_deep10m.py \
    sbatch data/deep10m/prepare_deep10m.sbatch
```

## Ground truth and IVF (LL)

When preparation has completed, from the project root:

```bash
sbatch scripts/ll_deep10m_a40.sbatch
```

This separate A40 job generates `deep10m_groundtruth.ivecs` (top-100 using
squared Euclidean distance over this 10M prefix) and builds IVF with 4096 lists.
Ground truth for the full billion-vector database is not interchangeable with
ground truth for this prefix. Data preparation alone does not submit this job.

The `data/` directory is excluded from Git. Version the preparation scripts
and this documentation, not the binary data files.
