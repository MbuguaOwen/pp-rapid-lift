import argparse, os, yaml, logging, numpy as np, torch, random, time
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.models.ts2vec import TSEncoder, ProjectionHead, nt_xent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOG = logging.getLogger("train_ts2vec")

def augment(x, mask_ratio=0.15, jitter_std=0.01):
    noise = torch.randn_like(x) * jitter_std
    xj = x + noise
    if mask_ratio>0:
        B,L,C = x.size()
        m = torch.rand(B,L,1, device=x.device) < mask_ratio
        xj = xj.masked_fill(m, 0.0)
    return xj

def resolve_device(want):
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return want

def main():
    p = argparse.ArgumentParser("Stage 3b — Train TS2Vec-style embeddings on micro windows")
    p.add_argument("--config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default=None)
    # Resume support: optional path or 'auto'
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="Resume from checkpoint path or 'auto' to use default")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config,"r"))
    root = cfg["io"]["windows_root"]; out_root = cfg["io"]["out_root"]; bar_type = cfg["io"]["bar_type"]
    epochs = int(cfg["train"]["epochs"]); bs = int(cfg["train"]["batch_size"]); lr = float(cfg["train"]["lr"])
    emb_dim = int(cfg["train"]["emb_dim"]); proj_dim = int(cfg["train"]["proj_dim"]); temp = float(cfg["train"]["temperature"])
    mask_ratio = float(cfg["train"]["mask_ratio"]); jitter_std = float(cfg["train"]["jitter_std"])
    want = args.device or cfg["train"].get("device","auto")
    dev = resolve_device(want)

    # Repro
    torch.manual_seed(13); np.random.seed(13); random.seed(13)

    # Load shards in memory-safe way: sample batches across shards
    roots = [os.path.join(root, sym, bar_type, ym) for sym in args.symbols for ym in args.months]
    roots = [r for r in roots if os.path.exists(os.path.join(r, "windows.npz"))]
    assert roots, "No windows found"

    # cache as memmaps (loads once, zero-copy reads thereafter)
    cache = {r: np.load(os.path.join(r, "windows.npz"), mmap_mode="r")["X"] for r in roots}
    keys = list(cache.keys())

    # infer channels
    sample = cache[keys[0]][0]  # (L,C)
    C = sample.shape[1]
    enc = TSEncoder(in_ch=C, emb_dim=emb_dim).to(dev)
    proj = ProjectionHead(emb_dim, proj_dim).to(dev)
    opt = torch.optim.Adam(list(enc.parameters())+list(proj.parameters()), lr=lr)
    scaler = GradScaler(enabled=(dev == "cuda"))

    # checkpoint paths
    os.makedirs(out_root, exist_ok=True)
    ckpt_path = os.path.join(out_root, f"ts2vec_ckpt_{bar_type}.pt")
    final_path = os.path.join(out_root, f"ts2vec_encoder_{bar_type}.pt")

    # --- RESUME ---
    start_ep = 1
    if args.resume:
        path = ckpt_path if args.resume in (True, "auto") else args.resume
        if os.path.exists(path):
            ck = torch.load(path, map_location=dev)
            enc.load_state_dict(ck["enc"])
            proj.load_state_dict(ck["proj"])
            opt.load_state_dict(ck["opt"])
            if "scaler" in ck:
                try:
                    scaler.load_state_dict(ck["scaler"])
                except Exception:
                    LOG.warning("Scaler state mismatch; continuing without scaler state")
            start_ep = int(ck.get("epoch", 0)) + 1
            # (optional) restore RNG states for exact reproducibility
            if "rng" in ck:
                try:
                    random.setstate(ck["rng"]["py"])
                    np.random.set_state(tuple(ck["rng"]["np"]))
                    torch.set_rng_state(ck["rng"]["torch"])
                except Exception:
                    LOG.warning("Could not restore RNG state; training will resume non-deterministically")
            LOG.info("Resumed from %s at epoch %d", path, start_ep)
        else:
            LOG.warning("--resume requested but no checkpoint at %s; starting fresh.", path)

    def sample_batch(bs):
        chunks = []
        need = bs
        while need > 0:
            r = random.choice(keys)
            X = cache[r]                 # (N,L,C) memmapped
            m = min(need, 128)           # take up to 128 from this shard
            j = np.random.randint(0, X.shape[0], size=m)
            chunks.append(torch.from_numpy(X[j]))   # (m,L,C)
            need -= m
        return torch.cat(chunks, dim=0).float()     # (B,L,C)

    try:
        for ep in range(start_ep, epochs+1):
            enc.train(); proj.train()
            losses = []
            steps = max(1, len(keys)) * 64  # was *128
            for _ in tqdm(range(steps), desc=f"epoch {ep}/{epochs}"):
                # sample a random batch across memmapped shards
                xb = sample_batch(bs).to(dev)                   # (B,L,C)
                with autocast(enabled=(dev == "cuda")):
                    x1 = augment(xb, mask_ratio, jitter_std)
                    x2 = augment(xb, mask_ratio, jitter_std)
                    z1 = proj(enc(x1)); z2 = proj(enc(x2))
                    loss = nt_xent(z1, z2, temperature=temp)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            LOG.info("epoch %d mean loss %.4f", ep, float(np.mean(losses)))

            # --- SAVE CHECKPOINT EACH EPOCH ---
            ck = {
                "epoch": ep,
                "time": time.time(),
                "enc": enc.state_dict(),
                "proj": proj.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()},
                "cfg": cfg,
            }
            torch.save(ck, ckpt_path)

        # save final encoder
        torch.save(enc.state_dict(), final_path)
        LOG.info("Saved encoder to %s", final_path)

    except KeyboardInterrupt:
        # save a last-ditch checkpoint on Ctrl+C
        ck = {
            "epoch": ep,
            "time": time.time(),
            "enc": enc.state_dict(),
            "proj": proj.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()},
            "cfg": cfg,
        }
        torch.save(ck, ckpt_path)
        LOG.info("Interrupted — checkpoint saved to %s", ckpt_path)

if __name__ == "__main__":
    main()
