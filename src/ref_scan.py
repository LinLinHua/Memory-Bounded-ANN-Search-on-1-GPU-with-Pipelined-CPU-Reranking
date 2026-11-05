"""Reference IVF scan in NumPy: candidate recall and reranked recall for 8-, 4-, and 1-bit codes.

For each query: probe the nprobe nearest lists, estimate distances from the codes,
keep the top-C candidates, then rerank them with exact distances.
Reports candidate recall (true top-10 found among the C candidates) and Recall@10
with and without exact reranking.
"""
import argparse, csv, json, os, time
import numpy as np


def read_vecs(path, dtype):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(dtype).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="directory with sift_*.fvecs/ivecs")
    p.add_argument("--index", required=True, help="output directory of build_ivf.py")
    p.add_argument("--nq", type=int, default=1000)
    p.add_argument("--csv", default="results/scan_recall_sift.csv")
    a = p.parse_args()

    xb = read_vecs(f"{a.data}/sift_base.fvecs", np.float32)
    xq = read_vecs(f"{a.data}/sift_query.fvecs", np.float32)[: a.nq]
    gt = read_vecs(f"{a.data}/sift_groundtruth.ivecs", np.int32)[: a.nq, :10]

    meta = json.load(open(f"{a.index}/meta.json"))
    n, d, L = meta["n"], meta["d"], meta["nlist"]
    load = lambda name, dt: np.fromfile(f"{a.index}/{name}", dtype=dt)
    cents = load("centroids.f32", np.float32).reshape(L, d)
    offs = load("list_offsets.i64", np.int64)
    ids = load("ids.i32", np.int32)
    vmin, step8, step4 = load("sq_min.f32", np.float32), load("sq_step8.f32", np.float32), load("sq_step4.f32", np.float32)
    rot = load("rotation.f32", np.float32).reshape(d, d)
    norms = load("norms.f32", np.float32)
    codes1 = load("codes1.u32", np.uint32).reshape(n, d // 32)

    # Decode 8- and 4-bit residuals once (list order)
    dec8 = vmin + load("codes8.u8", np.uint8).reshape(n, d).astype(np.float32) * step8
    c4 = load("codes4.u8", np.uint8).reshape(n, d // 2)
    q4 = np.empty((n, d), np.uint8)
    q4[:, 0::2], q4[:, 1::2] = c4 & 0xF, c4 >> 4
    dec4 = vmin + q4.astype(np.float32) * step4
    del c4, q4

    nprobes, Cs, Cmax = [16, 32, 64], [10, 50, 100, 200, 500, 1000], 1000
    rows_out = []
    for bits in (8, 4, 1):
        for nprobe in nprobes:
            t = time.time()
            cand_hit = np.zeros(len(Cs)); rr_hit = np.zeros(len(Cs)); norr_hit = 0.0
            for i in range(a.nq):
                q = xq[i]
                probe = np.argsort(((cents - q) ** 2).sum(1))[:nprobe]
                counts = offs[probe + 1] - offs[probe]
                rows = np.concatenate([np.arange(offs[l], offs[l + 1]) for l in probe])
                pos = np.repeat(np.arange(nprobe), counts)
                qr = q - cents[probe]                      # query residual per probed list
                if bits == 1:
                    qb = np.packbits((qr @ rot) > 0, axis=1, bitorder="little").view(np.uint32)
                    ham = np.bitwise_count(codes1[rows] ^ qb[pos]).sum(1)
                    qn = np.linalg.norm(qr, axis=1)[pos]
                    rn = norms[rows]
                    est = qn**2 + rn**2 - 2 * qn * rn * np.cos(np.pi * ham / d)
                else:
                    dec = dec8 if bits == 8 else dec4
                    est = ((qr[pos] - dec[rows]) ** 2).sum(1)
                k = min(Cmax, len(est))
                top = np.argpartition(est, k - 1)[:k]
                top = top[np.argsort(est[top])]
                cand = ids[rows[top]]
                g = set(gt[i])
                norr_hit += len(g & set(cand[:10])) / 10
                exact = ((xb[cand] - q) ** 2).sum(1)
                for j, C in enumerate(Cs):
                    c = cand[:C]
                    cand_hit[j] += len(g & set(c)) / 10
                    rr = c[np.argsort(exact[:C])[:10]]
                    rr_hit[j] += len(g & set(rr)) / 10
            dt = time.time() - t
            for j, C in enumerate(Cs):
                rows_out.append(dict(bits=bits, nprobe=nprobe, C=C,
                                     cand_recall=cand_hit[j] / a.nq,
                                     recall10_rerank=rr_hit[j] / a.nq,
                                     recall10_no_rerank=norr_hit / a.nq))
            print(f"bits={bits} nprobe={nprobe:3d}  no-rerank R@10={norr_hit / a.nq:.3f}  "
                  + "  ".join(f"C={C}:{rr_hit[j] / a.nq:.3f}" for j, C in enumerate(Cs))
                  + f"   ({dt:.0f} s)")

    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader(); w.writerows(rows_out)
    print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
