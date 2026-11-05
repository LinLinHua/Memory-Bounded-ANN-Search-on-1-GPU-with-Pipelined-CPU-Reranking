"""Build an IVF index for SIFT1M and encode residuals as 8-, 4-, and 1-bit codes.

Output files in --out (little-endian binary):
  centroids.f32 (L x d), list_offsets.i64 (L+1), ids.i32 (n, list order),
  codes8.u8 (n x d), codes4.u8 (n x d/2, low nibble first), codes1.u32 (n x d/32),
  norms.f32 (n), sq_min.f32 (d), sq_step8.f32 (d), sq_step4.f32 (d),
  rotation.f32 (d x d), meta.json
"""
import argparse, json, os, time
import numpy as np
import faiss


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def quantize(r, vmin, step, levels):
    return np.clip(np.rint((r - vmin) / step), 0, levels).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--train", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    x = read_fvecs(a.base)
    n, d = x.shape
    assert d % 32 == 0, "1-bit packing assumes d is a multiple of 32"
    print(f"base {n} x {d}")

    # k-means on a sample, then assign every vector to its nearest centroid
    t = time.time()
    sample = x[rng.choice(n, min(a.train, n), replace=False)]
    km = faiss.Kmeans(d, a.nlist, niter=20, seed=a.seed)
    km.train(sample)
    _, assign = km.index.search(x, 1)
    assign = assign.ravel()
    print(f"k-means + assignment: {time.time() - t:.1f} s")

    # Sort by list so each list is contiguous (CSR layout)
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=a.nlist)
    offsets = np.zeros(a.nlist + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    resid = x[order] - km.centroids[assign[order]]
    print(f"list size: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    # Scalar quantization with a shared per-dimension range
    vmin = resid.min(axis=0)
    vrange = np.maximum(resid.max(axis=0) - vmin, 1e-12)
    step8, step4 = vrange / 255.0, vrange / 15.0
    codes8 = quantize(resid, vmin, step8, 255)
    c4 = quantize(resid, vmin, step4, 15)
    codes4 = (c4[:, 0::2] | (c4[:, 1::2] << 4)).astype(np.uint8)

    # 1-bit codes: sign bits of residuals after a random orthogonal rotation
    rot, _ = np.linalg.qr(rng.standard_normal((d, d)))
    rot = rot.astype(np.float32)
    codes1 = np.packbits((resid @ rot) > 0, axis=1, bitorder="little").view(np.uint32)
    norms = np.linalg.norm(resid, axis=1).astype(np.float32)

    files = {
        "centroids.f32": km.centroids.astype(np.float32), "list_offsets.i64": offsets,
        "ids.i32": order.astype(np.int32), "codes8.u8": codes8, "codes4.u8": codes4,
        "codes1.u32": codes1, "norms.f32": norms, "sq_min.f32": vmin.astype(np.float32),
        "sq_step8.f32": step8.astype(np.float32), "sq_step4.f32": step4.astype(np.float32),
        "rotation.f32": rot,
    }
    for name, arr in files.items():
        np.ascontiguousarray(arr).tofile(os.path.join(a.out, name))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"n": n, "d": d, "nlist": a.nlist, "seed": a.seed}, f, indent=2)

    mb = lambda arr: arr.nbytes / 2**20
    print(f"index sizes (MB): 8-bit {mb(codes8):.0f}, 4-bit {mb(codes4):.0f}, "
          f"1-bit {mb(codes1):.0f}, raw FP32 {mb(x):.0f}")


if __name__ == "__main__":
    main()
