import sys, struct, numpy as np
eng = {}
d = open(sys.argv[1], 'rb').read(); p = 0
while p < len(d):
    nl = struct.unpack_from('<i', d, p)[0]; p += 4
    name = d[p:p+nl].decode(); p += nl
    nt, nv = struct.unpack_from('<ii', d, p); p += 8
    eng[name] = np.frombuffer(d, np.float32, nt*nv, p).reshape(nt, nv); p += 4*nt*nv
ref = np.load(sys.argv[2])
for name, L in eng.items():
    ti = ref[name + '.top_ids'][:, :64]; tv = ref[name + '.top_vals'][:, :64]; lse = ref[name + '.lse']
    kls, agree, nonfinite = [], 0, int((~np.isfinite(L)).sum())
    for t in range(L.shape[0]):
        lp_ref = tv[t] - lse[t]
        pr = np.exp(lp_ref.astype(np.float64)); prest = max(1.0 - pr.sum(), 1e-30)
        l = L[t].astype(np.float64); m = l.max(); le = m + np.log(np.exp(l - m).sum())
        lq = l[ti[t]] - le; q = np.exp(lq); qrest = max(1.0 - q.sum(), 1e-30)
        kls.append(float((pr * (lp_ref - lq)).sum() + prest * (np.log(prest) - np.log(qrest))))
        agree += int(np.argmax(L[t]) == ti[t, 0])
    print(f"{name}: T={L.shape[0]} meanKL={np.mean(kls):.3e} maxKL={np.max(kls):.3e} top1={agree / L.shape[0]:.3f} nonfinite={nonfinite}")
