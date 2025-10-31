import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix


# -----------------------------
# Focal Loss (drop-in for CE)
# -----------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5, reduction: str = "mean", ignore_index: Optional[int] = -100) -> None:
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            # Keep alpha on the right device and saved with the module
            self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            if not valid.any():
                return logits.new_tensor(0.0)
            logits = logits[valid]
            target = target[valid]

        ce = F.cross_entropy(logits, target, weight=getattr(self, "alpha", None), reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-9)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


# -----------------------------
# Utilities & Config
# -----------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def select_device(precision: str) -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = precision == "fp16"
        if use_amp:
            print("Device: CUDA (fp16 AMP enabled)")
        else:
            print("Device: CUDA (fp32)")
        return device, use_amp
    else:
        print("Device: CPU (AMP disabled; using fp32)")
        return torch.device("cpu"), False


def load_frame(path: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_parquet(path)
    dt = time.time() - t0
    print(f"Loaded DataFrame with {len(df):,} rows from {path} in {dt:.2f}s")
    return df


EXCLUDE = {
    "uid",
    "row_id",
    "fold",
    "t_first",
    "entry_ts",
    "t0",
    "t_last",
    "ts",
    "timestamp",
    "time",
    "symbol",
    # Explicitly exclude the target/mask columns from features
    "y_H240",
    "valid_H240",
}


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude label and mask families and explicit list
    feats: List[str] = []
    for c in numeric_cols:
        if c in EXCLUDE_COLS:
            continue
        if c.startswith("y_"):
            continue
        if c.startswith("valid_"):
            continue
        feats.append(c)
    return feats


_IMPUTE_WARNED = False


def _impute_inplace(x: np.ndarray) -> bool:
    # Impute NaN/Inf -> 0, return True if any were imputed
    global _IMPUTE_WARNED
    if x.size == 0:
        return False
    bad_mask = ~np.isfinite(x)
    if bad_mask.any():
        if not _IMPUTE_WARNED:
            warnings.warn(
                "Detected NaN/Inf in features; imputing with 0 (logged once)",
                RuntimeWarning,
            )
            _IMPUTE_WARNED = True
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return True
    return False


def build_feature_views(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # select numeric columns and drop excluded
    num = df.select_dtypes(include=["number"]).copy()
    keep = [c for c in num.columns if c not in EXCLUDE and not c.startswith("y_") and not c.startswith("valid_")]
    X = num[keep].to_numpy(dtype=np.float32)

    # labels (force strict int64)
    if "y_H240" not in df.columns:
        print("ERROR: 'y_H240' column not found in dataset.")
        sys.exit(1)
    y = df["y_H240"].astype("int64").to_numpy()

    # uid: force int64 (avoid uint64)
    if "uid" not in df.columns:
        print("ERROR: 'uid' column not found in dataset.")
        sys.exit(1)
    uid = df["uid"].astype("int64").to_numpy()

    # Impute non-finite in X
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, uid, keep


def detect_horizons(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"^y_[Hh]\d+$", str(c))]
    cols_sorted = sorted(cols, key=lambda s: int(re.findall(r"\d+", s)[0]))
    if not cols_sorted:
        print("ERROR: No horizon label columns found matching ^y_[Hh]\\d+$. ")
        sys.exit(1)
    return cols_sorted


def build_multi_targets(df: pd.DataFrame, horizons: List[str]) -> Tuple[np.ndarray, Dict[str, str]]:
    N = len(df)
    H = len(horizons)
    Y = np.full((N, H), -100, dtype=np.int64)
    mask_cols: Dict[str, str] = {}
    for i, h in enumerate(horizons):
        mask_col = "valid_" + h.split("_", 1)[1]
        if h not in df.columns:
            print(f"ERROR: Missing horizon label column: {h}")
            sys.exit(1)
        if mask_col not in df.columns:
            print(f"ERROR: Missing horizon valid mask column: {mask_col}")
            sys.exit(1)
        y_i = df[h].astype("Int64")
        m_i = df[mask_col].astype("int8").to_numpy()
        y_i = y_i.fillna(-100).astype("int64").to_numpy()
        y_i[m_i == 0] = -100
        Y[:, i] = y_i
        mask_cols[h] = mask_col
    return Y, mask_cols


class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, uid: np.ndarray) -> None:
        assert X.dtype == np.float32
        assert Y.dtype == np.int64
        assert uid.dtype == np.int64
        assert X.shape[0] == Y.shape[0] == uid.shape[0]
        assert Y.ndim == 2, "Y must be 2D: [N, H]"
        self.X = X
        self.Y = Y
        self.uid = uid

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # return proper tensor dtypes so default_collate can stack
        x = torch.from_numpy(self.X[i]).float()          # float32
        y = torch.from_numpy(self.Y[i]).long()           # int64/Long shape [H]
        u = torch.tensor(self.uid[i], dtype=torch.long)  # int64/Long
        return x, y, u


def safe_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, uids = zip(*batch)
    return (
        torch.stack(xs).float(),
        torch.stack(ys).long(),
        torch.stack(uids).long(),
    )


def _seed_worker(worker_id: int) -> None:
    # Ensure each worker has a deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


class MultiHeadMLP(nn.Module):
    def __init__(self, in_dim: int, n_heads: int, hidden: Sequence[int] = (512, 256)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.LayerNorm(h1),
            nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.LayerNorm(h2),
        )
        self.heads = nn.ModuleList([nn.Linear(h2, 3) for _ in range(n_heads)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)  # (B, hidden)
        outs = [head(z).unsqueeze(1) for head in self.heads]  # [(B,1,3)]*H
        return torch.cat(outs, dim=1)  # (B, H, 3)


def build_model(in_dim: int, horizons: List[str]) -> nn.Module:
    class _Model(nn.Module):
        def __init__(self, in_dim: int, horizons: List[str]):
            super().__init__()
            h1, h2 = 512, 256
            self.horizons = horizons
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.LayerNorm(h1),
                nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.LayerNorm(h2),
            )
            self.heads = nn.ModuleDict({h: nn.Linear(h2, 3) for h in horizons})

            def _init_weights(m: nn.Module) -> None:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

            self.apply(_init_weights)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            z = self.trunk(x)
            return {h: self.heads[h](z) for h in self.horizons}  # each (B,3)

    return _Model(in_dim, horizons)


def prepare_loss_map(df: pd.DataFrame, device: torch.device, horizons: List[str]) -> Dict[str, nn.Module]:
    # --- compute per-horizon weights from TRAIN fold only
    def class_weights_from(train_df: pd.DataFrame, y_col: str, mask_col: str) -> torch.Tensor:
        tr = (train_df["fold"] == 0) & (train_df[mask_col] == 1)
        y = (
            pd.to_numeric(train_df.loc[tr, y_col], errors="coerce")
            .astype("Int64")
            .dropna()
            .astype("int64")
        )
        cnt = y.value_counts().reindex([0, 1, 2]).fillna(0).to_numpy(dtype=float)
        cnt[cnt == 0] = 1.0
        w = (1.0 / cnt)
        w = w * (3.0 / w.sum())  # normalize to mean=1
        return torch.tensor(w, dtype=torch.float32, device=device)

    loss_map: Dict[str, nn.Module] = {}
    for h in horizons:
        mask_col = "valid_" + h.split("_", 1)[1]
        if mask_col in df.columns:
            w = class_weights_from(df, h, mask_col)
        else:
            # Fallback to uniform if mask column missing
            w = torch.ones(3, dtype=torch.float32, device=device)
        loss_map[h] = nn.CrossEntropyLoss(weight=w, label_smoothing=0.02, ignore_index=-100)
    return loss_map


def precision_at_k_tp(y_true: np.ndarray, p_tp: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(round(k_frac * n)))
    idx = np.argsort(-p_tp)[:k]
    return float((y_true[idx] == 2).mean())


@dataclass
class TrainState:
    best_pr_auc: float = -1.0
    best_epoch: int = -1
    epochs_no_improve: int = 0
    best_p10: float = 0.0
    best_p20: float = 0.0
    best_p30: float = 0.0


def first_batch_sanity(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_map: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
 ) -> None:
    model.train()
    xb, Yb, _ = next(iter(train_loader))  # Yb: (B, H)
    logits = model(xb.to(device))         # dict of (B,3)
    B, H = Yb.shape
    total_loss = 0.0
    nvalid = 0
    for i, h in enumerate(loss_map.keys()):
        Y_h = Yb[:, i].to(device)
        if (Y_h != -100).any():
            total_loss = total_loss + loss_map[h](logits[h], Y_h)
            nvalid += 1
    loss = total_loss / max(nvalid, 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"[sanity] xb{tuple(xb.shape)} yb{tuple(Yb.shape)} loss={float(loss):.4f}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_map: Dict[str, nn.Module],
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, Yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)              # dict of (B,3)
        total, nvalid = 0.0, 0
        for i, h in enumerate(loss_map.keys()):
            Y_h = Yb[:, i]
            if (Y_h != -100).any():
                total = total + loss_map[h](logits[h], Y_h)
                nvalid += 1
        loss = total / max(nvalid, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_map: Dict[str, nn.Module],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_probs: List[np.ndarray] = []   # (B, H, 3)
    all_y: List[np.ndarray] = []       # (B, H)
    all_uid: List[np.ndarray] = []     # (B,)
    total_loss = 0.0
    n_batches = 0
    horizons = list(loss_map.keys())
    for xb, Yb, uid in loader:
        xb = xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)
        logits_dict = model(xb)  # dict of (B,3)
        total, nvalid = 0.0, 0
        probs_stack: List[torch.Tensor] = []
        for i, h in enumerate(horizons):
            Y_h = Yb[:, i]
            if (Y_h != -100).any():
                total = total + loss_map[h](logits_dict[h], Y_h)
                nvalid += 1
            probs_stack.append(F.softmax(logits_dict[h], dim=-1).unsqueeze(1))
        loss = total / max(nvalid, 1)
        probs = torch.cat(probs_stack, dim=1)  # (B, H, 3)
        total_loss += float(loss.item())
        n_batches += 1
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(Yb.detach().cpu().numpy())
        all_uid.append(uid.detach().cpu().numpy())
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0, 3), dtype=np.float32)
    y_arr = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, 0), dtype=np.int64)
    uid_arr = np.concatenate(all_uid, axis=0) if all_uid else np.zeros((0,), dtype=np.int64)
    avg_loss = total_loss / max(1, n_batches)
    return {
        "loss": avg_loss,
        "probs": probs_arr,  # (N, H, 3)
        "Y": y_arr,          # (N, H)
        "uid": uid_arr,      # (N,)
    }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    # Binary collapse for PT (class 2)
    y_bin = (y_true == 2).astype(int)
    p_tp = probs[:, 2]
    base_rate = float((y_true == 2).mean()) if y_true.size > 0 else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pr_auc = float(average_precision_score(y_bin, p_tp))
        except Exception:
            pr_auc = 0.0
    p10 = precision_at_k_tp(y_true, p_tp, 0.10)
    p20 = precision_at_k_tp(y_true, p_tp, 0.20)
    p30 = precision_at_k_tp(y_true, p_tp, 0.30)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "pt_base_rate": base_rate,
        "pr_auc_tp": pr_auc,
        "p_at_10": float(p10),
        "p_at_20": float(p20),
        "p_at_30": float(p30),
        "confusion_matrix": cm.tolist(),
    }


def compute_metrics_per_horizon(horizons: List[str], Y: np.ndarray, probs: np.ndarray) -> Dict[str, Dict[str, Any]]:
    # Y: (N, H), probs: (N, H, 3)
    per_h: Dict[str, Dict[str, Any]] = {}
    N, H = Y.shape if Y.size else (0, len(horizons))
    for i, h in enumerate(horizons):
        mask = Y[:, i] != -100
        if mask.sum() == 0:
            per_h[h] = {
                "pt_base_rate": 0.0,
                "pr_auc_tp": 0.0,
                "p_at_10": 0.0,
                "p_at_20": 0.0,
                "p_at_30": 0.0,
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            }
            continue
        y_i = Y[mask, i]
        p_i = probs[mask, i, :]
        per_h[h] = compute_metrics(y_i, p_i)
    return per_h


def save_preds_csv(path: str, uid: np.ndarray, y_true: np.ndarray, probs: np.ndarray, fold: int) -> None:
    df = pd.DataFrame(
        {
            "uid": uid,
            "y_true": y_true,
            "p0": probs[:, 0],
            "p1": probs[:, 1],
            "p2": probs[:, 2],
            "fold": fold,
        }
    )
    df.to_csv(path, index=False)


def get_git_hash() -> Optional[str]:
    import subprocess

    try:
        h = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return h
    except Exception:
        return None


def cosine_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def write_run_config(
    outdir: str,
    args: argparse.Namespace,
    n_features: int,
    hidden: Sequence[int],
    out_dim: int,
) -> None:
    cfg = vars(args).copy()
    cfg.update(
        {
            "n_features": n_features,
            "hidden_dims": list(hidden),
            "out_dim": out_dim,
            "git_hash": get_git_hash(),
        }
    )
    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-horizon classifier")
    p.add_argument("--data", type=str, default="data/aligned/aligned_dollar_num.parquet")
    p.add_argument("--outdir", type=str, default="outputs/stage5_train")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=7, help="Early stop patience on PR-AUC")
    p.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])  # AMP only if CUDA
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    set_seed(args.seed)
    device, use_amp = select_device(args.precision)

    df = load_frame(args.data)
    # Detect horizons and build per-h targets/masks later
    horizons = detect_horizons(df)

    # --- compute per-horizon class weights from TRAIN only (fold==0) ---
    head_cols = [c for c in df.columns if c.startswith("y_")]        # ['y_H120','y_H240','y_H480']
    head_cols = sorted(head_cols, key=lambda s: int("".join(filter(str.isdigit, s))))
    cls_weights: Dict[str, torch.Tensor] = {}
    for h in head_cols:
        suf = h.split("_", 1)[1]                                      # e.g. 'H240'
        m = f"valid_{suf}"
        if m not in df.columns:
            continue
        tr = (df["fold"] == 0) & (df[m] == 1) & (df[h].isin([0, 1, 2]))
        counts = (
            df.loc[tr, h]
            .value_counts()
            .reindex([0, 1, 2], fill_value=0)
            .astype(float)
        )
        w = 1.0 / (counts + 1e-9)
        w = w / w.mean()
        w.iloc[2] = w.iloc[2] * 1.5   # small positive-class boost
        cls_weights[h] = torch.tensor(w.values, dtype=torch.float32, device=device)
    if "fold" not in df.columns:
        print("ERROR: 'fold' column not found in dataset.")
        sys.exit(1)
    # no requirement for a specific valid column now; per-h masks are used

    # Force safe dtypes after reading
    if "uid" in df.columns:
        df["uid"] = df["uid"].astype("int64")
    if "y_H240" in df.columns:
        df["y_H240"] = df["y_H240"].astype("int64")

    # Splits by fold; per-horizon masks are handled in the targets with ignore_index
    is_train = df["fold"] == 0
    is_val = df["fold"] == 1

    X_tr, _, uid_tr, feat_cols = build_feature_views(df.loc[is_train])
    Y_tr, _mask_cols_tr = build_multi_targets(df.loc[is_train], horizons)
    X_va, _, uid_va, _ = build_feature_views(df.loc[is_val])
    Y_va, _mask_cols_va = build_multi_targets(df.loc[is_val], horizons)
    n_features = len(feat_cols)

    # (Optional) Log the feature list so you can verify the count
    with open(os.path.join(args.outdir, "features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_cols))
    print(f"Features selected: {len(feat_cols)} (saved to features.txt)")

    write_run_config(
        outdir=args.outdir,
        args=args,
        n_features=n_features,
        hidden=(512, 256),
        out_dim=3,
    )

    # DataLoaders (for debugging, start with workers=0)
    train_ds = ParquetDataset(X_tr, Y_tr, uid_tr)
    val_ds = ParquetDataset(X_va, Y_va, uid_va)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
    )

    # Optional test split if present
    test_loader: Optional[torch.utils.data.DataLoader] = None
    if (df["fold"] == 2).any():
        is_test = df["fold"] == 2
        X_te, _, uid_te, _ = build_feature_views(df.loc[is_test])
        Y_te, _ = build_multi_targets(df.loc[is_test], horizons)
        test_ds = ParquetDataset(X_te, Y_te, uid_te)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=safe_collate,
        )

    model = build_model(n_features, horizons).to(device)
    loss_map = {
        h: FocalLoss(
            alpha=cls_weights.get(h, torch.ones(3, dtype=torch.float32, device=device)),
            gamma=1.5,
            reduction="mean",
            ignore_index=-100,
        )
        for h in horizons
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_scheduler(optimizer, args.epochs)

    # First-batch sanity check
    first_batch_sanity(model, train_loader, loss_map, optimizer, device, use_amp)

    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    best_path = os.path.join(args.outdir, "best.pt")
    last_path = os.path.join(args.outdir, "last.pt")

    state = TrainState()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_map, device, use_amp)
        val_out = evaluate(model, val_loader, loss_map, device)
        per_h_metrics = compute_metrics_per_horizon(horizons, val_out["Y"], val_out["probs"])

        # Early stopping on PR-AUC
        # Choose H240 for early stopping if available, else first horizon
        target_h = next((h for h in horizons if h.lower() == "y_h240"), horizons[0])
        pr_auc = per_h_metrics[target_h]["pr_auc_tp"]
        improved = pr_auc > state.best_pr_auc
        if improved:
            state.best_pr_auc = pr_auc
            state.best_epoch = epoch
            state.epochs_no_improve = 0
            state.best_p10 = per_h_metrics[target_h]["p_at_10"]
            state.best_p20 = per_h_metrics[target_h]["p_at_20"]
            state.best_p30 = per_h_metrics[target_h]["p_at_30"]
            torch.save(model.state_dict(), best_path)
        else:
            state.epochs_no_improve += 1

        # Always save last
        torch.save(model.state_dict(), last_path)

        # Save val predictions per horizon
        for i, h in enumerate(horizons):
            mask = val_out["Y"][:, i] != -100
            if mask.sum() == 0:
                continue
            y_i = val_out["Y"][mask, i]
            p_i = val_out["probs"][mask, i, :]
            val_csv = os.path.join(args.outdir, f"val_preds_{h.upper()}.csv")
            save_preds_csv(val_csv, val_out["uid"][mask], y_i, p_i, fold=1)

        # Log metrics
        current_lrs = [g["lr"] for g in optimizer.param_groups]
        rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_out["loss"]),
            "per_h_metrics": per_h_metrics,
            "lr": current_lrs[0] if current_lrs else None,
            "epoch_time_sec": float(time.time() - t0),
        }
        write_jsonl(metrics_path, rec)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_out['loss']:.6f}")
        for i, h in enumerate(horizons):
            m = per_h_metrics[h]
            print(
                f"[{h.upper()}] base={m['pt_base_rate']:.4f}  PR-AUC={m['pr_auc_tp']:.4f}  "
                f"P@10={m['p_at_10']:.4f}  P@20={m['p_at_20']:.4f}  P@30={m['p_at_30']:.4f}"
            )

        # Step scheduler after each epoch
        scheduler.step()

        # Early stopping check
        if state.epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)."
            )
            break

    # Summary
    print(
        f"Best epoch={state.best_epoch}, Best PR-AUC={state.best_pr_auc:.6f}, "
        f"P@10={state.best_p10:.4f}, P@20={state.best_p20:.4f}, P@30={state.best_p30:.4f}. "
        "See metrics.jsonl for per-epoch details."
    )

    # Optional: test-set evaluation if available
    if 'test_loader' in locals() and test_loader is not None:
        # Load best model for test eval
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
        except Exception:
            pass  # if missing, use current
        test_out = evaluate(model, test_loader, loss_map, device)
        test_metrics = compute_metrics_per_horizon(horizons, test_out["Y"], test_out["probs"]) 
        for i, h in enumerate(horizons):
            mask = test_out["Y"][:, i] != -100
            if mask.sum() == 0:
                continue
            y_i = test_out["Y"][mask, i]
            p_i = test_out["probs"][mask, i, :]
            test_csv = os.path.join(args.outdir, f"test_preds_{h.upper()}.csv")
            save_preds_csv(test_csv, test_out["uid"][mask], y_i, p_i, fold=2)
        print("Test metrics per horizon:")
        for h, m in test_metrics.items():
            print(
                f"[{h.upper()}] base={m['pt_base_rate']:.4f}  PR-AUC={m['pr_auc_tp']:.4f}  "
                f"P@10={m['p_at_10']:.4f}  P@20={m['p_at_20']:.4f}  P@30={m['p_at_30']:.4f}"
            )


if __name__ == "__main__":
    main()

# -----------------------------
# Future-proof note (no code)
# -----------------------------
# To extend this to true multi-horizon (e.g., y_H120, y_H240, y_H480):
# - Either add separate output heads per horizon, each masked by valid_Hxxx,
#   or train a shared trunk with stacked per-horizon logits.
# - Compute and report metrics (PR-AUC TP vs not-TP, P@k, confusion matrix)
#   per horizon separately.
# - Dataloaders would emit horizon-specific targets and masks; losses would be
#   aggregated across horizons using masks to ignore invalid samples for a given H.
