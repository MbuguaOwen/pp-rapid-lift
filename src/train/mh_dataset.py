import os, numpy as np, pandas as pd

def load_shards(aligned_root, embed_root, bar_type, symbols, months):
    rows = []
    for sym in symbols:
        for ym in months:
            a = os.path.join(aligned_root, sym, bar_type, ym, "aligned.parquet")
            e = os.path.join(embed_root, sym, bar_type, ym, "embed.npy")
            i = os.path.join(embed_root, sym, bar_type, ym, "index.parquet")
            if os.path.exists(a) and os.path.exists(e) and os.path.exists(i):
                rows.append((sym, ym, a, e, i))
    return rows

def assemble_matrix(shards, horizons, direction, use_regime_as, min_votes, regime_weight_alpha):
    X_list = []; Y_list = []; W_list = []; META_list = []
    for sym, ym, a, e, i in shards:
        D = pd.read_parquet(a)
        Z = np.load(e)
        I = pd.read_parquet(i)
        M = D.merge(I[["row_id"]], on="row_id", how="inner")
        idx = M.index.values
        X = Z[idx]

        Ys = []
        for H in horizons:
            y = M[f"y_H{H}"].copy()
            if direction=="up":
                y = (y==+1).astype(np.float32)
            elif direction=="down":
                y = (y==-1).astype(np.float32)
            elif direction=="trinary":
                y = y.map({-1:0,0:1,1:2}).astype(np.int64)
            Ys.append(y.to_numpy())
        Y = np.stack(Ys, axis=1)

        w = M["w"].to_numpy(dtype=np.float32)
        if use_regime_as=="filter":
            ok = (M["regime"].abs()>0) & (M[["bull_votes","bear_votes"]].max(axis=1)>=min_votes)
            X, Y, w, M = X[ok.values], Y[ok.values], w[ok.values], M.loc[ok]
        elif use_regime_as=="weight":
            w = w * (0.5 + float(regime_weight_alpha) * M["vote_ratio"].to_numpy(dtype=np.float32))

        X_list.append(X); Y_list.append(Y); W_list.append(w); META_list.append(M[["fold","is_train","is_valid"]])
    if not X_list: raise RuntimeError("No shards found")
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    W = np.concatenate(W_list, axis=0)
    META = pd.concat(META_list, ignore_index=True)
    return X, Y, W, META

def class_weights_per_horizon(Y, mask=None):
    H = Y.shape[1]
    cw = np.ones(H, dtype=np.float32)
    for h in range(H):
        y = Y[:,h]
        if mask is not None: y = y[mask]
        p = y.mean() if len(y)>0 else 0.5
        cw[h] = 1.0 / max(p, 1e-6)
    return (cw / cw.mean()).astype(np.float32)

