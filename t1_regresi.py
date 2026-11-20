"""Tugas 1 - Regresi: estimasi durasi_jam (P50 dan P90).
Jalankan: python t1_regresi.py   (mode cepat: QUICK=1)"""
import os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance, partial_dependence
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

QUICK = os.environ.get("QUICK") == "1"
NIT = 60 if QUICK else 350
tr = load("regresi_train"); te = load("regresi_test")
tr["ts"] = pd.to_datetime(tr.tgl_pickup); te["ts"] = pd.to_datetime(te.tgl_pickup)
tr["y"] = np.log(tr.durasi_jam)
R = {}

# ---------- (a) AUDIT ----------
SCAN_PROXY = [c for c in ["f04", "f05", "f22", "f23", "f17"]]
R["audit"] = dict(
    scan_mean_train=tr.jumlah_scan.mean(), scan_mean_test=te.jumlah_scan.mean(),
    corr_scan_logy=float(np.corrcoef(tr.jumlah_scan, tr.y)[0, 1]),
    corr_proxy_with_scan={c: float(np.corrcoef(tr[c], tr.jumlah_scan)[0, 1]) for c in SCAN_PROXY},
    jarak9999_train=int((tr.jarak_km == 9999).sum()), jarak0_train=int((tr.jarak_km == 0).sum()),
    jarak9999_test=int((te.jarak_km == 9999).sum()), berat_na_train=float(tr.berat_kg.isna().mean()),
    berat_na_test=float(te.berat_kg.isna().mean()), n_dur_gt500=int((tr.durasi_jam > 500).sum()),
    dur_max=float(tr.durasi_jam.max()), mitra_baru_test=sorted(set(te.mitra_kurir) - set(tr.mitra_kurir)),
    n_mitra_baru_test=int((~te.mitra_kurir.isin(tr.mitra_kurir)).sum()),
    f30_month_mean=tr.groupby(tr.tgl_pickup.str[:7]).f30.mean().round(3).to_dict(),
    f30_corr_y=float(np.corrcoef(tr.f30, tr.y)[0, 1]))
DROP = ["id_paket", "tgl_pickup", "ts", "jumlah_scan", "f30", "durasi_jam", "y"] + SCAN_PROXY

def feats(df):
    X = pd.DataFrame(index=df.index)
    j = df.jarak_km.where(~df.jarak_km.isin([0, 9999]))          # sentinel -> NaN
    X["jarak_miss"] = j.isna().astype(int); X["log_jarak"] = np.log1p(j)
    X["berat_miss"] = df.berat_kg.isna().astype(int); X["log_berat"] = np.log(df.berat_kg)
    X["log_vol"] = np.log(df.volume_liter); X["dens"] = np.log(df.berat_kg / df.volume_liter)
    X["log_nilai"] = np.log(df.nilai_barang); X["log_nilai_item"] = np.log(df.nilai_barang / df.jumlah_item)
    X["log_antrean"] = np.log1p(df.antrean_hub_asal); X["log_vol_penjual"] = np.log(df.volume_penjual_30h)
    X["log_umur_penjual"] = np.log(df.umur_akun_penjual_hari)
    X["jam"] = df.jam_pickup; X["dow"] = df.ts.dt.dayofweek; X["dom"] = df.ts.dt.day
    X["weekend"] = (X.dow >= 5).astype(int); X["malam"] = ((df.jam_pickup >= 20) | (df.jam_pickup < 5)).astype(int)
    X["log_hujan_a"] = np.log1p(df.hujan_asal_mm); X["log_hujan_t"] = np.log1p(df.hujan_tujuan_mm)
    for c in ["jumlah_item", "asuransi", "cod", "alamat_skor", "pengalaman_kurir_bulan", "rating_penjual", "indeks_kemacetan"]:
        X[c] = df[c]
    for c in [c for c in df.columns if c.startswith("f") and c[1:].isdigit() and c not in DROP]:
        X[c] = df[c]
    return X

CATS = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan"]
def cat_codes(df, ref):
    out = {}
    for c in CATS:
        lv = sorted(ref[c].unique()); m = {v: i for i, v in enumerate(lv)}
        out[c + "_c"] = df[c].map(m).astype(float)          # kategori tak dikenal (mis. mitra K13) -> NaN
    return pd.DataFrame(out, index=df.index)

def te_map(keys, y, m=30):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k").y.agg(["sum", "count"]); mu = y.mean()
    return ((g["sum"] + m * mu) / (g["count"] + m)).to_dict(), mu
def te_apply(keys, mp, mu): return keys.map(mp).fillna(mu).values
TE_COLS = {"kota_tujuan": lambda d: d.kota_tujuan, "rute": lambda d: d.kota_asal + ">" + d.kota_tujuan,
           "mitra_layanan": lambda d: d.mitra_kurir + d.layanan}

T0 = pd.Timestamp("2025-01-01")
def drift_feature(df, train, win=30, lag=7, minn=100):
    """Rata-rata residual log-durasi per mitra pada jendela 30 hari yang berakhir >=7 hari sebelum pickup
    (dan tidak melewati hari terakhir data latih). Menangkap perubahan rezim kurir (mis. K07 sejak Juli)."""
    kd = (train.ts.max() - T0).days; nd = kd + 1
    base = train.groupby(["layanan", "pulau_tujuan"]).y.transform("mean"); r = train.y - base
    mi = sorted(train.mitra_kurir.unique()); idx = {m: i for i, m in enumerate(mi)}
    S = np.zeros((len(mi), nd + 1)); N = np.zeros((len(mi), nd + 1))
    dd = (train.ts - T0).dt.days.values; mm = train.mitra_kurir.map(idx).values
    np.add.at(S, (mm, dd + 1), r.values); np.add.at(N, (mm, dd + 1), 1)
    S, N = S.cumsum(1), N.cumsum(1)
    d = (df.ts - T0).dt.days.values; m = df.mitra_kurir.map(idx)
    end = np.minimum(d - lag, kd) + 1; start = np.maximum(end - win, 0); ok = (m.notna().values) & (end > 0)
    mi_ = m.fillna(0).astype(int).values; e = np.clip(end, 0, nd); st = np.clip(start, 0, nd)
    ss = S[mi_, e] - S[mi_, st]; nn = N[mi_, e] - N[mi_, st]
    return np.where(ok & (nn >= minn), ss / np.maximum(nn, 1), np.nan)

def build(train, others, drift=True):
    """Fitur + target encoding (OOF untuk train, penuh untuk others). Tidak ada kebocoran label."""
    Xtr = pd.concat([feats(train), cat_codes(train, train)], axis=1)
    Xo = [pd.concat([feats(o), cat_codes(o, train)], axis=1) for o in others]
    if drift:
        Xtr["mitra_drift"] = drift_feature(train, train)
        for X, o in zip(Xo, others): X["mitra_drift"] = drift_feature(o, train)
    kf = KFold(5, shuffle=True, random_state=SEED)
    for name, fn in TE_COLS.items():
        k = fn(train); oof = np.zeros(len(train))
        for a, b in kf.split(train):
            mp, mu = te_map(k.iloc[a], train.y.iloc[a]); oof[b] = te_apply(k.iloc[b], mp, mu)
        Xtr["te_" + name] = oof
        mp, mu = te_map(k, train.y)
        for X, o in zip(Xo, others): X["te_" + name] = te_apply(fn(o), mp, mu)
    return Xtr, Xo

def hgb(q=None, **kw):
    p = dict(learning_rate=0.08, max_iter=NIT, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
             categorical_features="from_dtype", random_state=SEED)
    p.update(kw)
    if q is None: return HGB(loss="squared_error", **p)
    return HGB(loss="quantile", quantile=q, **p)
def catcols(X):  # tandai kolom kategori (kode numerik) lewat dtype category
    X = X.copy()
    for c in X.columns:
        if c.endswith("_c"): X[c] = X[c].astype("category") if X[c].notna().all() else X[c].astype("category")
    return X

def evaluate(y, p50, p90):
    return dict(pin50=pinball(y, p50, .5), pin90=pinball(y, p90, .9), cov90=float(np.mean(y <= p90)),
                width=float(np.mean(p90 - p50)))

# ---------- metode ----------
def m_baseline(train, Xtr, Xs, others):
    g = train.groupby(["layanan", "pulau_tujuan"]).durasi_jam
    q50, q90 = g.median().rename("a"), g.quantile(.9).rename("b")
    out = []
    for o in others:
        k = pd.MultiIndex.from_frame(o[["layanan", "pulau_tujuan"]])
        out.append((q50.reindex(k).fillna(train.durasi_jam.median()).values, q90.reindex(k).fillna(train.durasi_jam.quantile(.9)).values))
    return out

def m_linear(train, Xtr, Xs, others):
    """Ridge pada log(durasi) + kuantil residual heteroskedastik per layanan x pulau."""
    def prep(X, df, ref=None):
        Z = X.drop(columns=[c for c in X.columns if c.endswith("_c")]).copy()
        for c in ["layanan", "hub_asal", "pulau_tujuan", "mitra_kurir"]:
            for v in sorted((ref if ref is not None else train)[c].unique()): Z[f"{c}_{v}"] = (df[c] == v).astype(float)
        for v in sorted(train.layanan.unique()): Z[f"lj_{v}"] = (df.layanan == v) * X.log_jarak.fillna(X.log_jarak.median())
        Z["log_jarak"] = Z.log_jarak.fillna(train.pipe(lambda d: np.log1p(d.jarak_km.where(~d.jarak_km.isin([0, 9999]))).median()))
        Z["log_berat"] = Z.log_berat.fillna(0); Z["dens"] = Z.dens.fillna(0)
        if "mitra_drift" in Z: Z["drift_na"] = Z.mitra_drift.isna().astype(float); Z["mitra_drift"] = Z.mitra_drift.fillna(0)
        return Z
    Ztr = prep(Xtr, train); mu_, sd_ = Ztr.mean(), Ztr.std().replace(0, 1)
    mdl = Ridge(alpha=10.0).fit((Ztr - mu_) / sd_, train.y)
    res = train.y - mdl.predict((Ztr - mu_) / sd_)
    key = train.layanan + "|" + train.pulau_tujuan
    qs = {k: (np.quantile(r, .5), np.quantile(r, .9)) for k, r in res.groupby(key)}
    out = []
    for X, o in zip(Xs, others):
        mu = mdl.predict((prep(X, o) - mu_) / sd_)
        kk = (o.layanan + "|" + o.pulau_tujuan).map(qs); a = np.array([v[0] if isinstance(v, tuple) else np.quantile(res, .5) for v in kk])
        b = np.array([v[1] if isinstance(v, tuple) else np.quantile(res, .9) for v in kk])
        out.append((np.exp(mu + a), np.exp(mu + b)))
    return out

def m_hgb_quant(train, Xtr, Xs, others, params=None):
    params = params or {}
    Xt = catcols(Xtr); m5 = hgb(.5, **params).fit(Xt, train.y); m9 = hgb(.9, **params).fit(Xt, train.y)
    return [(np.exp(m5.predict(catcols(X))), np.exp(m9.predict(catcols(X)))) for X in Xs], (m5, m9)

def m_hgb_l2(train, Xtr, Xs, others):
    """HGB squared-error pada log(durasi) + kuantil residual per layanan (target transformation)."""
    Xt = catcols(Xtr); m = hgb(None).fit(Xt, train.y); res = train.y - m.predict(Xt)
    # residual in-sample bias kecil karena min_samples_leaf besar; kuantil per layanan
    qs = {k: (np.quantile(r, .5), np.quantile(r, .9)) for k, r in res.groupby(train.layanan)}
    out = []
    for X, o in zip(Xs, others):
        mu = m.predict(catcols(X)); a = o.layanan.map(lambda k: qs[k][0]).values; b = o.layanan.map(lambda k: qs[k][1]).values
        out.append((np.exp(mu + a), np.exp(mu + b)))
    return out

# ---------- (b) VALIDASI TEMPORAL vs ACAK ----------
def split(df, a, b): return df[df.ts < a], df[(df.ts >= a) & (df.ts < b)]
FOLDS = [("F1: latih Jan-Apr -> uji Mei-Jun", "2025-05-01", "2025-07-01"), ("F2: latih Jan-Jun -> uji Jul-Agu", "2025-07-01", "2025-09-01")]
results = {}; t0 = time.time()
for name, a, b in FOLDS:
    A, B = split(tr, pd.Timestamp(a), pd.Timestamp(b))
    Xa, (Xb,) = build(A, [B]); yb = B.durasi_jam.values; r = {}
    p = m_baseline(A, Xa, [Xb], [B])[0]; r["Baseline (median grup)"] = evaluate(yb, *p)
    p = m_linear(A, Xa, [Xb], [B])[0]; r["Linear log-target (Ridge)"] = evaluate(yb, *p)
    p = m_hgb_l2(A, Xa, [Xb], [B])[0]; r["HGB log-target + kuantil residual"] = evaluate(yb, *p)
    pp, _ = m_hgb_quant(A, Xa, [Xb], [B]); r["HGB quantile loss"] = evaluate(yb, *pp[0])
    results[name] = r; log(name, time.time() - t0); [log(" ", k, v) for k, v in r.items()]
R["temporal"] = results

# validasi acak 5-fold (dengan fitur bocor & tanpa), memakai model HGB quantile q=.5 saja agar cepat
rk = {}
Xall = pd.concat([feats(tr), cat_codes(tr, tr)], axis=1)
sub = tr.sample(120000, random_state=SEED) if not QUICK else tr.sample(40000, random_state=SEED)
def rand_vs_time(use_leak):
    Xs = Xall.loc[sub.index].copy()
    if use_leak:
        for c in ["jumlah_scan"] + SCAN_PROXY: Xs[c] = tr.loc[sub.index, c]
    ys = sub.y; out = {}
    idx = np.arange(len(sub)); kf = KFold(5, shuffle=True, random_state=SEED); ps = np.zeros(len(sub))
    for a, b in kf.split(idx):
        m = hgb(.5).fit(catcols(Xs.iloc[a]), ys.iloc[a]); ps[b] = m.predict(catcols(Xs.iloc[b]))
    out["random_cv_pin50"] = pinball(sub.durasi_jam.values, np.exp(ps), .5)
    msk = (sub.ts < "2025-07-01").values
    m = hgb(.5).fit(catcols(Xs[msk]), ys[msk]); pt = np.exp(m.predict(catcols(Xs[~msk])))
    out["temporal_pin50"] = pinball(sub.durasi_jam.values[~msk], pt, .5); return out
rk["tanpa_fitur_bocor"] = rand_vs_time(False); rk["dengan_fitur_bocor"] = rand_vs_time(True)
R["random_vs_temporal"] = rk; log(rk, time.time() - t0)
save_json(R, "t1_hasil_awal.json")

# ---------- (c) TUNING (hanya di dalam skema validasi temporal: fold F1) ----------
import itertools
A1, B1 = split(tr, pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")); Xa1, (Xb1,) = build(A1, [B1])
GRID = [dict(max_leaf_nodes=l, learning_rate=lr, min_samples_leaf=ms) for l, lr, ms in
        itertools.product([15, 31, 63], [0.05, 0.1], [100, 400])] if not QUICK else [dict(max_leaf_nodes=31, learning_rate=.1, min_samples_leaf=200)]
tune = []
for g in GRID:
    pp, _ = m_hgb_quant(A1, Xa1, [Xb1], [B1], g); e = evaluate(B1.durasi_jam.values, *pp[0]); e.update(g); tune.append(e)
    log("tune", g, round(e["pin50"], 3), round(e["pin90"], 3), round(e["cov90"], 3), time.time() - t0)
best = min(tune, key=lambda e: e["pin50"] + e["pin90"])
BP = {k: best[k] for k in ["max_leaf_nodes", "learning_rate", "min_samples_leaf"]}; R["tuning"] = tune; R["best_params"] = BP
log("best", BP)

# konfirmasi di F2 dengan parameter terbaik (+ ablasi fitur drift)
A2, B2 = split(tr, pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-01"))
XaF, (XbF,) = build(A2, [B2]); ppF, (m5F, m9F) = m_hgb_quant(A2, XaF, [XbF], [B2], BP); p50F, p90F = ppF[0]; yb2 = B2.durasi_jam.values
R["temporal"][FOLDS[1][0]]["HGB quantile loss (tuned + fitur drift)"] = evaluate(yb2, p50F, p90F)
Xa0, (Xb0,) = build(A2, [B2], drift=False); pp0, _ = m_hgb_quant(A2, Xa0, [Xb0], [B2], BP)
R["temporal"][FOLDS[1][0]]["HGB quantile loss (tuned, tanpa fitur drift)"] = evaluate(yb2, *pp0[0])
log("drift ablation", R["temporal"][FOLDS[1][0]]["HGB quantile loss (tuned + fitur drift)"], R["temporal"][FOLDS[1][0]]["HGB quantile loss (tuned, tanpa fitur drift)"])

# F3: latih Jan-Jun, kalibrasi Juli, uji Agustus -> bandingkan tanpa kalibrasi / konformal global / konformal per mitra (Mondrian)
A3a = tr[tr.ts < "2025-07-01"]; Jul = tr[(tr.ts >= "2025-07-01") & (tr.ts < "2025-08-01")]; Agu = tr[tr.ts >= "2025-08-01"]
X3a, (XJ, XG) = build(A3a, [Jul, Agu]); pp3, _ = m_hgb_quant(A3a, X3a, [XJ, XG], [Jul, Agu], BP)
(pj50, pj90), (pg50, pg90) = pp3; yj, yg = Jul.durasi_jam.values, Agu.durasi_jam.values
def offsets(df, y, p90, shrink=300):
    s_all = float(np.quantile(np.log(y) - np.log(p90), .9)); out = {}
    for k, d in pd.DataFrame({"m": df.mitra_kurir.values, "r": np.log(y) - np.log(p90)}).groupby("m"):
        n = len(d); out[k] = (n * float(np.quantile(d.r, .9)) + shrink * s_all) / (n + shrink)
    return s_all, out
s_all, s_m = offsets(Jul, yj, pj90)
g90 = pg90 * np.exp(s_all); m90 = pg90 * np.exp(Agu.mitra_kurir.map(s_m).fillna(s_all).values)
cmp_ = {"tanpa kalibrasi": evaluate(yg, pg50, pg90), "konformal global": evaluate(yg, pg50, np.maximum(g90, pg50)), "konformal per mitra (Mondrian)": evaluate(yg, pg50, np.maximum(m90, pg50))}
for nm, p in [("tanpa kalibrasi", pg90), ("konformal global", g90), ("konformal per mitra (Mondrian)", m90)]:
    cmp_[nm]["cov90_K07"] = float(np.mean(yg[(Agu.mitra_kurir == "K07").values] <= p[(Agu.mitra_kurir == "K07").values]))
    cmp_[nm]["cov90_non_K07"] = float(np.mean(yg[(Agu.mitra_kurir != "K07").values] <= p[(Agu.mitra_kurir != "K07").values]))
R["calibration"] = dict(offset_global_log=s_all, offset_per_mitra_log=s_m, evaluasi_agustus=cmp_, cov90_juli_sebelum=float(np.mean(yj <= pj90)))
log("kalibrasi", cmp_)
# tabel akhir F2 memakai prediksi tanpa kalibrasi + konformal per mitra dari F3 (offset Juli) untuk analisis Agu saja
p50F_, p90F_ = pg50, np.maximum(m90, pg50); B2 = Agu.copy(); yb2 = yg; p50F, p90Fc, m5F, XbF = pg50, p90F_, m5F, XbF
R["temporal"]["F3: latih Jan-Jun, kalibrasi Jul -> uji Agu"] = cmp_
mk = yb2 <= 500
R["outlier_impact"] = dict(n=int((~mk).sum()), pin50_all=pinball(yb2, p50F, .5), pin50_wo=pinball(yb2[mk], p50F[mk], .5),
                           pin90_all=pinball(yb2, p90Fc, .9), pin90_wo=pinball(yb2[mk], p90Fc[mk], .9),
                           cov90_all=float(np.mean(yb2 <= p90Fc)), cov90_wo=float(np.mean(yb2[mk] <= p90Fc[mk])))
XbF = XG  # fitur Agustus untuk interpretasi (model F3 kuantil 0.5)
_, (m5F, m9F) = m_hgb_quant(A3a, X3a, [XG], [Agu], BP)

# ---------- (f) ANALISIS ERROR per layanan / pulau / mitra ----------
B2 = B2.copy(); B2["p50"], B2["p90"] = p50F, p90Fc
def grp(col):
    g = B2.groupby(col); rows = {}
    for k, d in g:
        rows[k] = dict(n=len(d), pin50=pinball(d.durasi_jam.values, d.p50.values, .5), pin90=pinball(d.durasi_jam.values, d.p90.values, .9),
                       cov90=float(np.mean(d.durasi_jam <= d.p90)), median_aktual=float(d.durasi_jam.median()),
                       bias50=float(np.mean(d.durasi_jam <= d.p50)))
    return rows
R["error_by"] = {c: grp(c) for c in ["layanan", "pulau_tujuan", "mitra_kurir"]}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, c in zip(ax, ["layanan", "pulau_tujuan", "mitra_kurir"]):
    d = pd.DataFrame(R["error_by"][c]).T; a.bar(range(len(d)), d.cov90, color="#4a7"); a.axhline(.9, color="r", ls="--", lw=1)
    a.set_xticks(range(len(d))); a.set_xticklabels(d.index, rotation=60, ha="right", fontsize=7); a.set_ylim(.7, 1); a.set_title(f"Cakupan P90 per {c}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t1_cakupan.png"), dpi=150); plt.close()

# interpretasi: permutation importance + partial dependence (model F2 kuantil 0.5)
rs = np.random.RandomState(SEED); ix = rs.choice(len(XbF), min(10000, len(XbF)), replace=False)
Xs_, ys_ = catcols(XbF.iloc[ix]), np.log(yb2[ix])
class Wrap:  # skor pinball skala log agar permutation_importance dapat dipakai
    def __init__(s, m): s.m = m
    def fit(s, *a): return s
    def predict(s, X): return s.m.predict(X)
from sklearn.metrics import make_scorer
sc = make_scorer(lambda y, p: -pinball(y, p, .5))
pi = permutation_importance(m5F, Xs_, ys_, scoring=sc, n_repeats=3, random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=Xs_.columns).sort_values(ascending=False); R["perm_importance"] = imp.head(15).round(5).to_dict()
fig, ax = plt.subplots(figsize=(6, 4.2)); imp.head(15)[::-1].plot.barh(ax=ax, color="#36c"); ax.set_title("Permutation importance (kenaikan pinball q=0.5, skala log)", fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_t1_importance.png"), dpi=150); plt.close()
top_num = [c for c in imp.index if not c.endswith("_c")][:4]
fig, ax = plt.subplots(1, len(top_num), figsize=(3.2 * len(top_num), 3))
for a, c in zip(np.atleast_1d(ax), top_num):
    pdp = partial_dependence(m5F, Xs_, [c], grid_resolution=25); a.plot(pdp["grid_values"][0], np.exp(pdp["average"][0])); a.set_title(c, fontsize=9); a.set_ylabel("P50 (jam)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t1_pdp.png"), dpi=150); plt.close()

# ---------- Model final: latih semua data latih; offset konformal per mitra dari model Jan-Jul dikalibrasi pada Agustus ----------
A4 = tr[tr.ts < "2025-08-01"]; Aug = tr[tr.ts >= "2025-08-01"]
X4, (XA,) = build(A4, [Aug]); pp4, _ = m_hgb_quant(A4, X4, [XA], [Aug], BP)
s_all4, s_m4 = offsets(Aug, Aug.durasi_jam.values, pp4[0][1]); R["offset_final"] = dict(global_log=s_all4, per_mitra_log=s_m4)
Xf, (Xt,) = build(tr, [te]); ppf, _ = m_hgb_quant(tr, Xf, [Xt], [te], BP); p50t, p90t = ppf[0]
p90t = np.maximum(p90t * np.exp(te.mitra_kurir.map(s_m4).fillna(s_all4).values), p50t * 1.001)
sub_ = pd.DataFrame({"id_paket": te.id_paket, "p50": np.round(p50t, 2), "p90": np.round(p90t, 2)})
assert sub_.p90.ge(sub_.p50).all() and sub_.id_paket.is_unique and len(sub_) == len(te) and sub_.notna().all().all()
sub_.to_csv(os.path.join(ROOT, "prediksi_regresi.csv"), index=False)
R["pred_summary"] = sub_[["p50", "p90"]].describe().round(2).to_dict()
R["pred_by_mitra"] = pd.DataFrame({"m": te.mitra_kurir, "p50": p50t, "p90": p90t}).groupby("m").median().round(1).to_dict("index")
save_json(R, "t1_hasil.json"); log("SELESAI T1", time.time() - t0)
