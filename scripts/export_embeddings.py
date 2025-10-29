import argparse, os, yaml, numpy as np, torch, pandas as pd
from tqdm import tqdm
from src.models.ts2vec import TSEncoder

def main():
    p = argparse.ArgumentParser("Stage 3c — Export embeddings for all windows")
    p.add_argument("--config", required=True)
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config,"r"))
    wr = cfg["io"]["windows_root"]; out_root = cfg["io"]["out_root"]; bt = cfg["io"]["bar_type"]
    emb_path = os.path.join(out_root, f"ts2vec_encoder_{bt}.pt")
    assert os.path.exists(emb_path), "Train encoder first"
    sample = np.load(os.path.join(wr, args.symbols[0], bt, args.months[0], "windows.npz"))["X"][0]
    C = sample.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = TSEncoder(in_ch=C, emb_dim=int(cfg["train"]["emb_dim"])).to(dev)
    enc.load_state_dict(torch.load(emb_path, map_location=dev))
    enc.eval()

    for sym in args.symbols:
        for ym in tqdm(args.months, desc="embed", unit="month"):
            in_dir = os.path.join(wr, sym, bt, ym)
            if not os.path.exists(os.path.join(in_dir,"windows.npz")): continue
            out_dir = os.path.join(out_root, sym, bt, ym)
            os.makedirs(out_dir, exist_ok=True)
            X = np.load(os.path.join(in_dir,"windows.npz"))["X"]    # (N,L,C)
            idx = pd.read_parquet(os.path.join(in_dir,"index.parquet"))
            Z_list = []
            B = 2048
            for i in range(0, len(X), B):
                with torch.no_grad():
                    Z = enc(torch.from_numpy(X[i:i+B]).to(dev).float()).cpu().numpy()
                    Z_list.append(Z)
            Z = np.vstack(Z_list)
            np.save(os.path.join(out_dir, "embed.npy"), Z)
            idx.to_parquet(os.path.join(out_dir, "index.parquet"))

if __name__ == "__main__":
    main()

