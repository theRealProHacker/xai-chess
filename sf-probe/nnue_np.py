#!/usr/bin/env python3
"""numpy reader for the xai-chess NNUE dump (magic XAINNUE1, 10780-byte records).

The record is a fixed-size C struct with no padding, so numpy can describe it as a
structured dtype and memory-map the file. Nothing is copied until you touch a field,
so a 1.6 GB dump costs no RAM to open and each layer is a plain 2-D array.

    from nnue_np import load, LAYERS
    d = load("dump.bin")            # memmap, no read
    d["l1"]                         # (N, 1024) uint8  -- first nonlinearity
    d["acc"][:, 0, :]               # (N, 1024) int16  -- accumulator, side to move
    LAYERS["l1"]                    # ('nonlinear', 1024, 'clip + pairwise product')
"""
import numpy as np

MAGIC = b"XAINNUE1"
L1, FC0, AC0, FC1, MAX_KA, MAX_TH = 1024, 16, 30, 32, 32, 128
HEADER = 24
OUTPUT_SCALE, WEIGHT_SCALE_BITS = 16, 6

# Field order matches DumpRecord in nnue_misc.cpp: widest-first, padding-free.
DTYPE = np.dtype([
    ("psqt",        "<i4"),
    ("positional",  "<i4"),
    ("simple_eval", "<i4"),
    ("fc2",         "<i4"),
    ("idx_halfka",  "<i4", (2, MAX_KA)),
    ("idx_threats", "<i4", (2, MAX_TH)),
    ("fc0",         "<i4", (FC0,)),
    ("fc1",         "<i4", (FC1,)),
    ("acc",         "<i2", (2, L1)),
    ("tacc",        "<i2", (2, L1)),
    ("n_halfka",    "<u2", (2,)),
    ("n_threats",   "<u2", (2,)),
    ("l1",          "u1",  (L1,)),
    ("ac0",         "u1",  (AC0,)),
    ("ac1",         "u1",  (FC1,)),
    ("stm",         "u1"),
    ("bucket",      "u1"),
    ("in_check",    "u1"),
    ("small_net",   "u1"),
    ("_pad",        "u1",  (2,)),
])
assert DTYPE.itemsize == 10780, DTYPE.itemsize

# Where each probe target sits in the forward pass, and whether the map from the
# INPUT to it is linear. Only the nonlinear ones can exceed a linear input probe.
# IMPORTANT -- l1 ordering. On a vectorised build the transformer writes its output
# through vec_packus_16, which interleaves per 128-bit lane. So l1[j] is NOT the pair
# (acc[j], acc[j+512]); it is that product under a FIXED permutation of the 512 pair
# indices, per perspective. Verified: the multiset of l1 equals the multiset of
# (clip(acc+tacc)_lo * clip(acc+tacc)_hi)//512 exactly, but elementwise agreement is
# only ~83%. For probing this is harmless -- a fixed permutation of input dimensions
# changes no probe's accuracy. It matters only if you want to map an individual l1
# unit back to accumulator units; recover the permutation empirically from a few
# thousand records (value diversity disambiguates it) before doing that.
LAYERS = {
    "acc":  ("linear",    2 * L1, "accumulation: sum of weight columns + bias"),
    "tacc": ("linear",    2 * L1, "threatAccumulation: same, over FullThreats"),
    "l1":   ("nonlinear", L1,     "clip(acc+tacc) then pairwise product /512"),
    "fc0":  ("nonlinear", FC0,    "affine on l1; index 15 is the skip term"),
    "ac0":  ("nonlinear", AC0,    "ac_sqr_0 || ac_0 (squared-clipped and clipped ReLU)"),
    "fc1":  ("nonlinear", FC1,    "affine on ac0"),
    "ac1":  ("nonlinear", FC1,    "clipped ReLU"),
    "fc2":  ("nonlinear", 1,      "affine to the scalar eval"),
}


def load(path, mmap=True):
    with open(path, "rb") as f:
        head = f.read(HEADER)
    if head[:8] != MAGIC:
        raise ValueError(f"bad magic {head[:8]!r}")
    rec_size = int(np.frombuffer(head, "<u4", 1, 12)[0])
    count = int(np.frombuffer(head, "<u8", 1, 16)[0])
    if rec_size != DTYPE.itemsize:
        raise ValueError(f"record size {rec_size} != {DTYPE.itemsize}; reader is stale")
    mode = "r" if mmap else None
    a = (np.memmap(path, dtype=DTYPE, mode="r", offset=HEADER, shape=(count,)) if mmap
         else np.fromfile(path, dtype=DTYPE, count=count, offset=HEADER))
    return a


def recompute_positional(a):
    """Vectorised reproduction of propagate()'s return + Network::evaluate's scaling.
    C++ integer division truncates toward zero; numpy // floors. Hence the sign dance."""
    def tdiv(num, den):
        q = np.abs(num) // den
        return np.where(num < 0, -q, q)
    fwd = tdiv(a["fc0"][:, FC0 - 1].astype(np.int64) * (600 * OUTPUT_SCALE),
               127 * (1 << WEIGHT_SCALE_BITS))
    return tdiv(a["fc2"].astype(np.int64) + fwd, OUTPUT_SCALE)


def probe_matrix(a, layer, perspective="stm"):
    """Flatten one layer to (N, D) float32, ready for a probe."""
    x = a[layer]
    if x.ndim == 3:                      # (N, 2, D) -- acc / tacc
        x = x[:, 0, :] if perspective == "stm" else (
            x[:, 1, :] if perspective == "opp" else x.reshape(len(x), -1))
    elif x.ndim == 1:
        x = x[:, None]
    return np.ascontiguousarray(x, dtype=np.float32)


if __name__ == "__main__":
    import sys
    a = load(sys.argv[1])
    got, want = recompute_positional(a), a["positional"]
    print(f"records: {len(a)}   dtype itemsize: {DTYPE.itemsize}")
    print(f"positional recomputed from fc0/fc2: {(got == want).sum()}/{len(a)} match")
    print(f"buckets seen: {sorted(set(a['bucket'].tolist()))}   in_check: {int(a['in_check'].sum())}"
          f"   small_net_would_run: {int(a['small_net'].sum())}")
    print()
    print(f"{'layer':6} {'kind':10} {'shape':>14}  description")
    for name, (kind, dim, desc) in LAYERS.items():
        print(f"{name:6} {kind:10} {str(probe_matrix(a, name).shape):>14}  {desc}")
