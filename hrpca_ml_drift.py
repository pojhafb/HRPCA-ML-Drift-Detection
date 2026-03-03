#!/usr/bin/env python3
"""
ELEC2 harmful drift evaluation + structural subspace monitoring + rank sweep.

What this script does
---------------------
1) Loads ELEC2 (OpenML electricity v1), encodes label UP=1, DOWN=0.
2) Builds model-facing features X1 = [raw features, rolling mean, rolling std] with window ROLL_L1.
3) Splits time-ordered data into train / val / prod (fractions).
4) Trains LogisticRegression on train.
5) Slides fixed-size windows over train (for threshold calibration) and prod (for evaluation).
6) Computes drift metrics per window:
   - Structural: D_fro (subspace deviation), R_subspace (residual-out-of-subspace)
   - Model-aware: proj_mean (mean shift along beta), projE (projection energy shift)
   - Baselines: PSI(avg), KS(frac p<alpha), MMD(RBF)
7) Defines "harmful" windows as top-q AUC-drop windows in prod.
8) Summarizes harmful recall / FPR / TP / FP for each detector.
9) Produces paper-ready plots:
   - residual_vs_auc (colored harmful/benign + threshold line)
   - projection_vs_auc (projE vs AUC drop)
   - rank_sweep_tradeoff (FPR vs harmful recall across ranks)

Outputs
-------
Single-rank run:
  {prefix}_results_summary_rank{r}.csv
  {prefix}_residual_vs_auc_rank{r}.pdf
  {prefix}_projection_vs_auc_rank{r}.pdf
  {prefix}_residual_roc_rank{r}.pdf   (optional, enabled by default)
  {prefix}_debug_windows_rank{r}.csv  (optional debug dump)

Rank sweep:
  {prefix}_rank_sweep.csv
  {prefix}_rank_sweep_tradeoff.pdf

Usage examples
--------------
  python elec_realdata_eval_full.py --rank 2
  python elec_realdata_eval_full.py --rank-sweep --ranks 2,4,6,8,10
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


# -----------------------------
# Defaults / Config
# -----------------------------
SEED = 42
np.random.seed(SEED)

TRAIN_FRAC = 0.60
VAL_FRAC = 0.10  # kept, though calibration uses train windows by default

WINDOW_SIZE = 384
WINDOW_STRIDE = 96

ROLL_L1 = 24  # rolling feature window size (time steps)

# harmful windows: top-q AUC-drop windows in production
HARMFUL_Q = 0.80

# thresholds: mean + k*std on baseline windows
CALIB_K_DEFAULT = 4.0
CALIB_K_STRUCT_DEFAULT = 2.0

# KS baseline uses fraction of features with p < alpha
KS_ALPHA = 0.01

# MMD subsampling for speed
MMD_MAX_POINTS = 2000

# Output defaults
DEFAULT_PREFIX = "elec"


# -----------------------------
# Math / Drift utilities
# -----------------------------
def unit(v: np.ndarray) -> np.ndarray:
    nv = norm(v)
    return v / nv if nv > 0 else v


def top_r_projection_cov(X: np.ndarray, rank: int) -> np.ndarray:
    """
    Projection onto top-r eigenspace of mean-centered covariance.
    X is window data (n x d).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0])
    _, V = np.linalg.eigh(C)
    U = V[:, -rank:]
    return U @ U.T


def subspace_deviation_fro(Pa: np.ndarray, Pb: np.ndarray) -> float:
    return float(norm(Pa - Pb, "fro"))


def residual_out_of_subspace(W: np.ndarray, P_ref: np.ndarray) -> float:
    resid = W - (W @ P_ref)
    return float(norm(resid, "fro") / (norm(W, "fro") + 1e-9))


def psi_featurewise(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    PSI averaged over features. Quantile bins from ref.
    """
    psis = []
    for j in range(ref.shape[1]):
        x_ref = ref[:, j]
        x_cur = cur[:, j]

        qs = np.quantile(x_ref, np.linspace(0, 1, bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9

        ref_hist, _ = np.histogram(x_ref, bins=qs)
        cur_hist, _ = np.histogram(x_cur, bins=qs)

        ref_p = ref_hist / max(1, ref_hist.sum())
        cur_p = cur_hist / max(1, cur_hist.sum())

        ref_p = np.clip(ref_p, eps, 1.0)
        cur_p = np.clip(cur_p, eps, 1.0)

        psis.append(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return float(np.mean(psis)) if psis else 0.0


def ks_fraction_below(ref: np.ndarray, cur: np.ndarray, alpha: float = KS_ALPHA) -> float:
    """
    KS test per feature; returns fraction of features with p < alpha.
    """
    ps = [ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(ref.shape[1])]
    ps = np.asarray(ps, dtype=float)
    return float(np.mean(ps < alpha))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None, max_points: int = MMD_MAX_POINTS) -> float:
    """
    Unbiased-ish MMD^2 with RBF kernel; subsamples for speed.
    """
    rng = np.random.default_rng(SEED)
    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], size=max_points, replace=False)]
    if Y.shape[0] > max_points:
        Y = Y[rng.choice(Y.shape[0], size=max_points, replace=False)]

    if gamma is None:
        Z = np.vstack([X, Y])
        idx = rng.choice(Z.shape[0], size=min(800, Z.shape[0]), replace=False)
        Zs = Z[idx]
        dists = []
        for i in range(min(200, Zs.shape[0] - 1)):
            diff = Zs[i + 1:] - Zs[i]
            d2 = np.sum(diff * diff, axis=1)
            dists.append(d2)
        dists = np.concatenate(dists) if len(dists) else np.array([1.0])
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / med

    def k_rbf(A, B):
        a2 = np.sum(A * A, axis=1).reshape(-1, 1)
        b2 = np.sum(B * B, axis=1).reshape(1, -1)
        d2 = a2 + b2 - 2.0 * (A @ B.T)
        return np.exp(-gamma * d2)

    Kxx = k_rbf(X, X)
    Kyy = k_rbf(Y, Y)
    Kxy = k_rbf(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    m = X.shape[0]
    n = Y.shape[0]
    mmd2 = (Kxx.sum() / (m * (m - 1) + 1e-9)) + (Kyy.sum() / (n * (n - 1) + 1e-9)) - 2.0 * Kxy.mean()
    return float(mmd2)


def hi_thr(s: pd.Series, k: float) -> float:
    return float(s.mean() + k * s.std(ddof=0))


# -----------------------------
# Data loader + feature builder
# -----------------------------
def load_elec2():
    """
    Load ELEC2 from OpenML (electricity v1).
    Fallback to elec2.csv in cwd if OpenML fails.
    """
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name="electricity", version=1, as_frame=True)
        df = ds.frame.copy()
        print("[INFO] OpenML loaded dataset name=electricity version=1")

        target_col = "class" if "class" in df.columns else ds.target
        y = df[target_col].astype(str).str.strip().str.upper()
        y = (y == "UP").astype(int)
        y = pd.Series(y).astype(int)

        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).astype(float)

        print("[INFO] label counts:", y.value_counts().to_dict())
        return X.reset_index(drop=True), y.reset_index(drop=True)
    except Exception as e:
        print("[WARN] OpenML load failed:", e)

    csv_path = "elec2.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "class" not in df.columns:
            raise RuntimeError("elec2.csv must have 'class' column with UP/DOWN labels.")
        y = df["class"].astype(str).str.strip().str.upper()
        y = (y == "UP").astype(int)
        y = pd.Series(y).astype(int)
        X = df.drop(columns=["class"]).select_dtypes(include=[np.number]).astype(float)
        print("[INFO] loaded elec2.csv label counts:", y.value_counts().to_dict())
        return X.reset_index(drop=True), y.reset_index(drop=True)

    raise RuntimeError("Could not load ELEC2. Use OpenML or place elec2.csv in cwd.")


def add_rolling_features(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Model-facing features:
      - raw features
      - rolling mean per feature
      - rolling std per feature
    """
    df = X.copy()
    roll_mean = df.rolling(window=window, min_periods=window).mean().add_suffix(f"_rmean{window}")
    roll_std = df.rolling(window=window, min_periods=window).std().fillna(0).add_suffix(f"_rstd{window}")
    out = pd.concat([df, roll_mean, roll_std], axis=1).dropna().reset_index(drop=True)
    return out


# -----------------------------
# Plot helpers (paper-ready)
# -----------------------------
def plot_residual_vs_auc(df_prod: pd.DataFrame, thr_R: float, out_pdf: str, title: str):
    harmful = df_prod["harmful"].values.astype(bool)
    x = df_prod["R_subspace"].values
    y = df_prod["auc_drop"].values

    plt.figure(figsize=(6, 5))
    plt.scatter(x[~harmful], y[~harmful], s=18, label="Benign windows")
    plt.scatter(x[harmful], y[harmful], s=22, label="Harmful windows (top-q AUC drop)")
    plt.axvline(thr_R, linestyle="--", linewidth=2, label=r"Threshold $\tau_R$")
    plt.xlabel(r"Residual-out-of-subspace $R=\|W-WP_{ref}\|_F/\|W\|_F$")
    plt.ylabel("AUC drop")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def plot_projection_vs_auc(df_prod: pd.DataFrame, out_pdf: str, title: str):
    x = df_prod["projE"].values
    y = df_prod["auc_drop"].values

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=18)
    plt.xlabel(r"Projection energy drift $|\Delta \mathbb{E}[(\beta^\top x)^2]|$")
    plt.ylabel("AUC drop")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def plot_roc_for_metric(df_prod: pd.DataFrame, metric_col: str, out_pdf: str, title: str):
    y_true = df_prod["harmful"].values.astype(int)
    scores = df_prod[metric_col].values

    ts = np.quantile(scores, np.linspace(0.0, 1.0, 101))
    tprs, fprs = [], []
    for t in ts:
        y_pred = (scores > t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        tprs.append(tpr)
        fprs.append(fpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fprs, tprs, marker="o", markersize=3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Harmful Recall)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def plot_rank_sweep_tradeoff(df_sweep: pd.DataFrame, out_pdf: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.scatter(df_sweep["Rsub_fpr"], df_sweep["Rsub_recall"], s=60)
    for _, row in df_sweep.iterrows():
        plt.text(row["Rsub_fpr"] + 0.002, row["Rsub_recall"] + 0.005, f"r={int(row['rank'])}", fontsize=9)
    plt.xlabel("FPR (Subspace Residual)")
    plt.ylabel("Harmful Recall (Subspace Residual)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Core evaluation (single rank)
# -----------------------------
def run_for_rank(
    rank: int,
    calib_k: float,
    calib_k_struct: float,
    out_prefix: str = DEFAULT_PREFIX,
    dump_windows: bool = False,
    make_roc: bool = True,
):
    X0, y0 = load_elec2()
    print("Loaded ELEC2:", X0.shape)

    X0 = X0.fillna(0.0)

    # Build model-facing features
    X1 = add_rolling_features(X0, ROLL_L1)
    y1 = y0.iloc[len(y0) - len(X1):].reset_index(drop=True)

    X = X1
    y = y1

    print("[INFO] Overall y counts:", np.bincount(y.values))

    n = len(X)
    n_train = int(TRAIN_FRAC * n)
    n_val = int((TRAIN_FRAC + VAL_FRAC) * n)

    X_train_raw = X.iloc[:n_train].values
    y_train = y.iloc[:n_train].values

    X_val_raw = X.iloc[n_train:n_val].values
    y_val = y.iloc[n_train:n_val].values

    X_prod_raw = X.iloc[n_val:].values
    y_prod_all = y.iloc[n_val:].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_prod = scaler.transform(X_prod_raw)

    print("[INFO] y_train class counts:", np.bincount(y_train))

    model = LogisticRegression(max_iter=3000, n_jobs=1)
    model.fit(X_train, y_train)

    baseline_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    print("Baseline train AUC:", baseline_auc)

    ref = X_train
    P_ref = top_r_projection_cov(ref, rank)

    beta_hat = unit(model.coef_.reshape(-1))
    z_ref = ref @ beta_hat

    def sliding_metrics(X_stream: np.ndarray, y_stream: np.ndarray) -> pd.DataFrame:
        rows = []
        for start in range(0, max(1, X_stream.shape[0] - WINDOW_SIZE + 1), WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            W = X_stream[start:end]
            yw = y_stream[start:end]
            if len(W) < WINDOW_SIZE:
                continue

            P_w = top_r_projection_cov(W, rank)
            D_fro = subspace_deviation_fro(P_ref, P_w)

            R_sub = residual_out_of_subspace(W, P_ref)

            auc = roc_auc_score(yw, model.predict_proba(W)[:, 1])
            auc_drop = baseline_auc - auc

            proj = float(abs(beta_hat @ (W.mean(axis=0) - ref.mean(axis=0))))

            z_w = W @ beta_hat
            projE = float(abs((z_w ** 2).mean() - (z_ref ** 2).mean()))

            psi = psi_featurewise(ref, W)
            ks_frac = ks_fraction_below(ref, W, alpha=KS_ALPHA)
            mmd = mmd_rbf(ref, W)

            rows.append({
                "start": start,
                "end": end,
                "D_fro": D_fro,
                "R_subspace": R_sub,
                "auc_drop": auc_drop,
                "proj": proj,
                "projE": projE,
                "psi": psi,
                "ks_frac": ks_frac,
                "mmd": mmd,
            })
        return pd.DataFrame(rows)

    # --- calibration windows on TRAIN ---
    df_cal = sliding_metrics(X_train, y_train)
    if df_cal.empty:
        raise RuntimeError("Calibration windows empty. Reduce WINDOW_SIZE/STRIDE or check dataset length.")

    thr = {
        "D_fro": hi_thr(df_cal["D_fro"], k=calib_k_struct),
        "R_subspace": hi_thr(df_cal["R_subspace"], k=calib_k_struct),

        "proj": hi_thr(df_cal["proj"], k=calib_k),
        "projE": hi_thr(df_cal["projE"], k=calib_k),
        "psi": hi_thr(df_cal["psi"], k=calib_k),
        "mmd": hi_thr(df_cal["mmd"], k=calib_k),
        "ks_frac": hi_thr(df_cal["ks_frac"], k=calib_k),
    }
    print(f"Calibrated thresholds (train windows, mean+{calib_k_struct}std structural, mean+{calib_k}std others): {thr}")

    # --- production windows ---
    df_prod = sliding_metrics(X_prod, y_prod_all)
    if df_prod.empty:
        raise RuntimeError("Production windows empty. Reduce WINDOW_SIZE/STRIDE or check dataset length.")

    q = float(np.quantile(df_prod["auc_drop"], HARMFUL_Q))
    df_prod["harmful"] = (df_prod["auc_drop"] >= q).astype(int)
    print(f"[INFO] harmful threshold (AUC drop, q={HARMFUL_Q:.2f}) = {q} positives = {int(df_prod['harmful'].sum())} out of {len(df_prod)}")

    # flags
    df_prod["flag_psi"] = (df_prod["psi"] > thr["psi"]).astype(int)
    df_prod["flag_ks"] = (df_prod["ks_frac"] > thr["ks_frac"]).astype(int)
    df_prod["flag_mmd"] = (df_prod["mmd"] > thr["mmd"]).astype(int)
    df_prod["flag_proj"] = (df_prod["proj"] > thr["proj"]).astype(int)
    df_prod["flag_projE"] = (df_prod["projE"] > thr["projE"]).astype(int)
    df_prod["flag_Dfro"] = (df_prod["D_fro"] > thr["D_fro"]).astype(int)
    df_prod["flag_Rsub"] = (df_prod["R_subspace"] > thr["R_subspace"]).astype(int)

    def summarize(flag_col: str):
        y_true = df_prod["harmful"].values
        y_pred = df_prod[flag_col].values
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        recall = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        return recall, fpr, tp, fp

    methods = [
        ("PSI(avg)", "flag_psi"),
        ("KS(frac p<0.01)", "flag_ks"),
        ("MMD(RBF)", "flag_mmd"),
        ("Proj mean", "flag_proj"),
        ("Proj energy", "flag_projE"),
        ("Subspace Dk (Fro)", "flag_Dfro"),
        ("Subspace Residual", "flag_Rsub"),
    ]

    rows = []
    for name, col in methods:
        recall, fpr, tp, fp = summarize(col)
        rows.append({"Method": name, "HarmfulRecall": recall, "FPR": fpr, "TP": tp, "FP": fp})
    summary = pd.DataFrame(rows)

    # correlations
    corr_D = float(np.corrcoef(df_prod["D_fro"], df_prod["auc_drop"])[0, 1])
    corr_R = float(np.corrcoef(df_prod["R_subspace"], df_prod["auc_drop"])[0, 1])
    corr_proj = float(np.corrcoef(df_prod["proj"], df_prod["auc_drop"])[0, 1])
    corr_projE = float(np.corrcoef(df_prod["projE"], df_prod["auc_drop"])[0, 1])

    # write summary
    summary_csv = f"{out_prefix}_results_summary_rank{rank}.csv"
    summary.to_csv(summary_csv, index=False)
    print("\nWrote:", summary_csv)
    print(summary)

    print("\nCorrelations on prod windows:")
    print(" corr(D_fro, AUC drop)       =", corr_D)
    print(" corr(R_subspace, AUC drop)  =", corr_R)
    print(" corr(proj_mean, AUC drop)   =", corr_proj)
    print(" corr(proj_energy, AUC drop) =", corr_projE)

    # optional windows dump (debug / appendix)
    if dump_windows:
        dump_csv = f"{out_prefix}_debug_windows_rank{rank}.csv"
        df_prod.to_csv(dump_csv, index=False)
        print("Wrote window-level debug CSV:", dump_csv)

    # plots for paper
    fig_resid = f"{out_prefix}_residual_vs_auc_rank{rank}.pdf"
    fig_proj = f"{out_prefix}_projection_vs_auc_rank{rank}.pdf"
    plot_residual_vs_auc(
        df_prod=df_prod,
        thr_R=float(thr["R_subspace"]),
        out_pdf=fig_resid,
        title=f"ELEC2: Structural Residual vs AUC Drop (rank={rank})"
    )
    plot_projection_vs_auc(
        df_prod=df_prod,
        out_pdf=fig_proj,
        title=f"ELEC2: Model-aware Projection Energy vs AUC Drop"
    )
    print("\nWrote figures:")
    print(" -", fig_resid)
    print(" -", fig_proj)

    if make_roc:
        fig_roc = f"{out_prefix}_residual_roc_rank{rank}.pdf"
        plot_roc_for_metric(
            df_prod=df_prod,
            metric_col="R_subspace",
            out_pdf=fig_roc,
            title=f"ELEC2: ROC for Structural Residual (rank={rank})"
        )
        print(" -", fig_roc)

    # Package key outputs for sweep
    get_row = lambda m, c: summary.loc[summary["Method"] == m, c].iloc[0]
    return {
        "rank": rank,
        "thr_D_fro": float(thr["D_fro"]),
        "thr_R_subspace": float(thr["R_subspace"]),
        "corr_D_fro": corr_D,
        "corr_R_subspace": corr_R,
        "corr_proj": corr_proj,
        "corr_projE": corr_projE,
        "Rsub_recall": float(get_row("Subspace Residual", "HarmfulRecall")),
        "Rsub_fpr": float(get_row("Subspace Residual", "FPR")),
        "Rsub_tp": int(get_row("Subspace Residual", "TP")),
        "Rsub_fp": int(get_row("Subspace Residual", "FP")),
        "Dfro_recall": float(get_row("Subspace Dk (Fro)", "HarmfulRecall")),
        "Dfro_fpr": float(get_row("Subspace Dk (Fro)", "FPR")),
    }


# -----------------------------
# Rank sweep
# -----------------------------
def run_rank_sweep(ranks, calib_k, calib_k_struct, out_prefix=DEFAULT_PREFIX):
    rows = []
    for r in ranks:
        print("\n" + "=" * 90)
        print(f"[RANK SWEEP] Running rank={r}")
        print("=" * 90)
        res = run_for_rank(
            rank=r,
            calib_k=calib_k,
            calib_k_struct=calib_k_struct,
            out_prefix=out_prefix,
            dump_windows=False,
            make_roc=False,
        )
        rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = f"{out_prefix}_rank_sweep.csv"
    df.to_csv(out_csv, index=False)
    print("\nWrote rank sweep CSV:", out_csv)
    print(df[["rank", "Rsub_recall", "Rsub_fpr", "corr_R_subspace", "Dfro_recall", "Dfro_fpr", "corr_D_fro"]])

    out_pdf = f"{out_prefix}_rank_sweep_tradeoff.pdf"
    plot_rank_sweep_tradeoff(
        df_sweep=df,
        out_pdf=out_pdf,
        title="ELEC2: Rank Sweep Trade-off (Structural Residual)"
    )
    print("Wrote rank sweep plot:", out_pdf)
    return df


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=2, help="Single-rank run (default: 2)")
    p.add_argument("--rank-sweep", action="store_true", help="Run rank sweep and write sweep CSV/plot")
    p.add_argument("--ranks", type=str, default="2,4,6,8,10", help="Comma-separated ranks for sweep")
    p.add_argument("--calib-k", type=float, default=CALIB_K_DEFAULT,
                   help="k for non-struct thresholds: mean + k*std (default: 4.0)")
    p.add_argument("--calib-k-struct", type=float, default=CALIB_K_STRUCT_DEFAULT,
                   help="k for structural thresholds: mean + k*std (default: 2.0)")
    p.add_argument("--harmful-q", type=float, default=HARMFUL_Q,
                   help="Quantile for harmful windows by AUC drop (default: 0.80 => top 20%)")
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    p.add_argument("--out-prefix", type=str, default=DEFAULT_PREFIX)
    p.add_argument("--dump-windows", action="store_true", help="Write per-window prod metrics CSV (debug)")
    p.add_argument("--no-roc", action="store_true", help="Disable ROC plot generation")
    return p.parse_args()


def main():
    global HARMFUL_Q, WINDOW_SIZE, WINDOW_STRIDE
    args = parse_args()

    HARMFUL_Q = float(args.harmful_q)
    WINDOW_SIZE = int(args.window_size)
    WINDOW_STRIDE = int(args.window_stride)

    if args.rank_sweep:
        ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
        run_rank_sweep(ranks=ranks, calib_k=args.calib_k, calib_k_struct=args.calib_k_struct, out_prefix=args.out_prefix)
    else:
        run_for_rank(
            rank=args.rank,
            calib_k=args.calib_k,
            calib_k_struct=args.calib_k_struct,
            out_prefix=args.out_prefix,
            dump_windows=args.dump_windows,
            make_roc=(not args.no_roc),
        )


if __name__ == "__main__":
    main()
