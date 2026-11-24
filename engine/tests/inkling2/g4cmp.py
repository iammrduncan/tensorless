import sys, struct, numpy as np
def load(f):
    out = {}; d = open(f, 'rb').read(); p = 0
    while p < len(d):
        nl = struct.unpack_from('<i', d, p)[0]; p += 4
        name = d[p:p+nl].decode(); p += nl
        nt, nv = struct.unpack_from('<ii', d, p); p += 8
        out[name] = np.frombuffer(d, np.float32, nt*nv, p).reshape(nt, nv); p += 4*nt*nv
    return out
A, B = load(sys.argv[1]), load(sys.argv[2])
ref = np.load(sys.argv[3])
def lsm(x):
    x = x.astype(np.float64); m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
for k in A:
    a, b = lsm(A[k]), lsm(B[k])
    kl = (np.exp(a) * (a - b)).sum(-1)
    print(f"{k}: prefill-vs-decode meanKL={kl.mean():.3e} maxKL={kl.max():.3e} argmax-agree={(a.argmax(-1) == b.argmax(-1)).mean():.3f}; decode top1 vs ref={(b.argmax(-1) == ref[k + '.top_ids'][:, 0]).mean():.3f}")
