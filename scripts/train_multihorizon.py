import os, yaml, numpy as np, torch, torch.nn as nn, random
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from src.train.mh_dataset import load_shards, assemble_matrix, class_weights_per_horizon

def device_of(d):
    if d=="auto": return "cuda" if torch.cuda.is_available() else "cpu"
    return d

class MHHead(nn.Module):
    def __init__(self, emb_dim, H, hidden=0):
        super().__init__()
        if hidden>0:
            self.f = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, H))
        else:
            self.f = nn.Linear(emb_dim, H)
    def forward(self, x): return self.f(x)

def precision_at_topk(y_true, y_score, frac):
    n = max(1, int(len(y_true)*frac))
    idx = np.argsort(-y_score)[:n]
    return float((y_true[idx].sum() / n)) if n>0 else 0.0

def main():
    cfg = yaml.safe_load(open("configs/train_multihorizon.yaml","r"))
    dev = device_of(cfg["train"]["device"])
    torch.manual_seed(int(cfg["train"].get("seed",13))); np.random.seed(int(cfg["train"].get("seed",13))); random.seed(int(cfg["train"].get("seed",13)))

    shards = load_shards(cfg["io"]["aligned_root"], cfg["io"]["embed_root"], cfg["io"]["bar_type"], cfg["io"]["symbols"], cfg["io"]["months"])
    X, Y, W, META = assemble_matrix(
        shards,
        horizons=cfg["task"]["horizons"],
        direction=cfg["task"]["direction"],
        use_regime_as=cfg["task"]["use_regime_as"],
        min_votes=cfg["task"]["min_votes"],
        regime_weight_alpha=float(cfg["task"]["regime_weight_alpha"])
    )
    
    folds = np.sort(META["fold"].unique())
    H = Y.shape[1]; emb_dim = X.shape[1]
    all_fold_metrics = []

    for k in folds:
        is_tr = META["is_train"].to_numpy().astype(bool) & (META["fold"].to_numpy()!=k)
        is_va = META["is_valid"].to_numpy().astype(bool) & (META["fold"].to_numpy()==k)

        Xtr, Ytr, Wtr = X[is_tr], Y[is_tr], W[is_tr]
        Xva, Yva, Wva = X[is_va], Y[is_va], W[is_va]

        cw = torch.tensor(class_weights_per_horizon(Ytr), dtype=torch.float32, device=dev)

        model = MHHead(emb_dim, H, hidden=int(cfg["model"]["head_hidden"])).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        B = int(cfg["train"]["batch_size"])
        for ep in range(1, int(cfg["train"]["epochs"])+1):
            model.train()
            idx = np.random.permutation(len(Xtr))
            for i in range(0, len(Xtr), B):
                j = idx[i:i+B]
                xb = torch.from_numpy(Xtr[j]).to(dev).float()
                yb = torch.from_numpy(Ytr[j]).to(dev).float()
                wb = torch.from_numpy(Wtr[j]).to(dev).float()
                logits = model(xb)
                loss_mat = loss_fn(logits, yb)      # (B,H)
                loss = (loss_mat * cw).mean(dim=1)
                loss = (loss * wb).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xva).to(dev).float()).cpu().numpy()
            probs  = 1/(1+np.exp(-logits))
        metrics = {}
        for frac in cfg["train"]["topk"]:
            for h_idx, Hh in enumerate(cfg["task"]["horizons"]):
                p_at_k = precision_at_topk(Yva[:,h_idx], probs[:,h_idx], frac)
                metrics[f"P@{int(frac*100)}%_H{Hh}"] = p_at_k
        for h_idx, Hh in enumerate(cfg["task"]["horizons"]):
            try:
                ap = average_precision_score(Yva[:,h_idx], probs[:,h_idx])
            except Exception:
                ap = float("nan")
            metrics[f"PR_AUC_H{Hh}"] = float(ap)
        all_fold_metrics.append(metrics)

    keys = sorted(all_fold_metrics[0].keys())
    avg = {k: float(np.nanmean([m[k] for m in all_fold_metrics])) for k in keys}
    print("=== Multi-Horizon CV (fold-avg) ===")
    for k in keys:
        print(f"{k}: {avg[k]:.4f}")

if __name__ == "__main__":
    main()

