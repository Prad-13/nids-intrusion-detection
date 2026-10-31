"""
Network Intrusion Detection - GDG on Campus AI/ML recruitment challenge
Final solution. Reproduces the submitted files:

  submission_bag.csv   regularized, bagged LightGBM                      (public 0.84486)
  submission_pl.csv    same model + pseudo-labeling                      (public 0.84475, private 0.99019)
  submission_augA.csv  bag + reply-masked augmentation (the explained fix) (public 0.84335)

Usage:
  python solution.py --data-dir /path/to/folder/with/csvs --out-dir out
  python solution.py --check          # also prints the two sanity checks used in the README
  python solution.py --group-cv       # also runs the (slow) pseudo-profile group-CV

Only the provided competition data is used. No labels of the test set are used anywhere.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
DEFAULT_DIR = "/kaggle/input/competitions/network-intrusion-detection2-ml-challenge/"


# ----------------------------------------------------------------------------- data
def load_data(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = tr.pop("Target").values
    ids = te["id"]
    tr = tr.drop(columns=["id"], errors="ignore")
    te = te.drop(columns=["id"], errors="ignore")
    return tr, te, y, ids


def encode_and_engineer(tr, te):
    """Shared category codes for proto/service (train+test) and a few scale-free features."""
    tr, te = tr.copy(), te.copy()
    for c in ["proto", "service"]:
        cats = pd.concat([tr[c], te[c]]).astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], cats).codes
        te[c] = pd.Categorical(te[c], cats).codes

    def fe(d):
        d = d.copy()
        d["tot_pkts"] = d["fwd_pkts_tot"] + d["bwd_pkts_tot"]
        d["pkt_ratio"] = d["fwd_pkts_tot"] / (d["bwd_pkts_tot"] + 1)
        d["syn_frac"] = d["flow_SYN_flag_count"] / (d["tot_pkts"] + 1)
        return d.replace([np.inf, -np.inf], np.nan).fillna(0)

    return fe(tr), fe(te)


# ----------------------------------------------------------------------------- model
def make_bag(seed, mcs=100, col=0.5):
    """Heavily regularized LightGBM: extra-trees splits, L2, row/column subsampling,
    large min_child_samples, balanced class weights (macro F1 weights all classes equally)."""
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=mcs, subsample=0.7, subsample_freq=1, colsample_bytree=col,
        reg_lambda=5, extra_trees=True, class_weight="balanced",
        random_state=seed, verbose=-1, n_jobs=-1)


def bagged_proba(X, y, Xt, n_seeds):
    P = None
    for s in range(n_seeds):
        p = make_bag(s).fit(X, y).predict_proba(Xt)
        P = p / n_seeds if P is None else P + p / n_seeds
    return P


# ----------------------------------------------------------------------------- reply-masking fix
ZERO_KW = ("iat", "active", "idle", "per_sec", "per_second", "duration", "down_up")


def mask_reply(d):
    """Neutralize every feature that only exists because a reply came back
    (backward packets, RST/ACK/FIN flags, duration, rates, timing)."""
    d = d.copy()
    for c in d.columns:
        if c.startswith("bwd") or any(k in c for k in ZERO_KW):
            d[c] = 0
    for c in ["flow_RST_flag_count", "flow_ACK_flag_count", "flow_FIN_flag_count",
              "flow_ECE_flag_count", "flow_CWR_flag_count"]:
        d[c] = 0
    for s in ["min", "max", "tot", "avg"]:
        d["flow_pkts_payload." + s] = d["fwd_pkts_payload." + s]
    d["flow_pkts_payload.std"] = 0
    d["tot_pkts"] = d["fwd_pkts_tot"]
    d["pkt_ratio"] = d["fwd_pkts_tot"]
    d["syn_frac"] = d["flow_SYN_flag_count"] / (d["tot_pkts"] + 1)
    return d


def build_aug(X, y, frac=0.2, seed=1):
    """Add reply-masked copies of single-packet Availability flows that did get a reply."""
    rng = np.random.default_rng(seed)
    q = ((X.fwd_pkts_tot == 1) & (X.bwd_pkts_tot >= 1)).values & (y == "Availability")
    keep = q & (rng.random(len(y)) < frac)
    return pd.concat([X, mask_reply(X[keep])], ignore_index=True), np.r_[y, y[keep]]


# ----------------------------------------------------------------------------- checks
def sanity_checks(tr, te, y, Xb, classes):
    from sklearn.model_selection import train_test_split
    sig = lambda d: (d["fwd_pkts_payload.tot"] == 120) & (d["fwd_init_window_size"] == 64)
    print("\n[check 1] hping-style signature (payload 120 & window 64)")
    print("  test rows with signature:", int(sig(te).sum()), f"({sig(te).mean():.1%} of test)",
          "| of which with no reply:", int((sig(te) & (te.bwd_pkts_tot == 0)).sum()))
    print("  train classes with signature:", pd.Series(y[sig(tr).values]).value_counts().to_dict())

    print("[check 2] Availability recall on reply-masked held-out train rows")
    a, b = train_test_split(np.arange(len(Xb)), test_size=0.2, random_state=0, stratify=y)
    Xv, yv = Xb.iloc[b], y[b]
    sel = ((Xv.fwd_pkts_tot == 1) & (Xv.bwd_pkts_tot >= 1)).values
    Xv_m, yv_m = mask_reply(Xv[sel]), yv[sel]
    for name, (X_, y_) in {"no augmentation": (Xb.iloc[a], y[a]),
                           "with augmentation": build_aug(Xb.iloc[a], y[a])}.items():
        p = make_bag(0).fit(X_, y_).predict(Xv_m)
        print(f"  {name:18s}: {(p[yv_m == 'Availability'] == 'Availability').mean():.3f}")


def group_cv(Xb, y, classes):
    """Pseudo-profile validation: KMeans clusters inside each class, GroupKFold on clusters.
    Harsher than the real shift; used only for RELATIVE comparisons."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    Z = StandardScaler().fit_transform(np.log1p(Xb.select_dtypes("number").clip(lower=0)))
    groups = np.zeros(len(Xb), int)
    for i, c in enumerate(classes):
        m = y == c
        groups[m] = KMeans(min(5, m.sum()), n_init=3, random_state=0).fit_predict(Z[m]) + 10 * i
    oof = np.empty(len(Xb), object)
    for a, b in GroupKFold(5).split(Xb, y, groups):
        oof[b] = make_bag(0).fit(Xb.iloc[a], y[a]).predict(Xb.iloc[b])
    print(f"\n[group-CV] single regularized LightGBM, macro F1 = {f1_score(y, oof, average='macro'):.4f}"
          "  (harsh proxy; use for relative comparison only)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--group-cv", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tr_raw, te_raw, y, ids = load_data(args.data_dir)
    tr, te = encode_and_engineer(tr_raw, te_raw)
    classes = np.unique(y)
    print("train", tr.shape, "test", te.shape)
    print("train class share:", pd.Series(y).value_counts(normalize=True).round(3).to_dict())

    # id.orig_p looks like an ephemeral client port (a capture fingerprint) -> dropped
    Xb, Xtb = tr.drop(columns=["id.orig_p"]), te.drop(columns=["id.orig_p"])

    if args.check:
        sanity_checks(tr, te, y, Xb, classes)
    if args.group_cv:
        group_cv(Xb, y, classes)

    def save(name, P):
        pred = classes[P.argmax(1)]
        pd.DataFrame({"id": ids, "Target": pred}).to_csv(os.path.join(args.out_dir, name), index=False)
        print(f"{name}: {pd.Series(pred).value_counts(normalize=True).round(3).to_dict()}")
        return pred

    # 1) regularized bagged LightGBM
    P_bag = bagged_proba(Xb, y, Xtb, args.seeds)
    pred_bag = save("submission_bag.csv", P_bag)

    # 2) pseudo-labeling: add test rows the bag is >90% confident about, labeled by the bag itself
    conf = P_bag.max(1) > 0.9
    print(f"pseudo-labeling: {conf.mean():.1%} of test rows used")
    Xpl = pd.concat([Xb, Xtb[conf]], ignore_index=True)
    ypl = np.r_[y, pred_bag[conf]]
    P_pl = bagged_proba(Xpl, ypl, Xtb, args.seeds)
    save("submission_pl.csv", P_pl)

    # 3) reply-masked augmentation (the principled fix for the reply-confounded labels)
    Xaug, yaug = build_aug(Xb, y)
    P_aug = bagged_proba(Xaug, yaug, Xtb, args.seeds)
    save("submission_augA.csv", P_aug)


if __name__ == "__main__":
    main()
