"""Tugas 2 - Klasifikasi fraud: probabilitas is_fraud + keputusan flag berbasis biaya.
Jalankan: python t2_klasifikasi.py   (QUICK=1 untuk uji cepat)"""
import os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

QUICK = os.environ.get("QUICK") == "1"; NIT = 40 if QUICK else 150; t0 = time.time()
COST_REVIEW = 25000.0
tr = load("klasifikasi_train"); te = load("klasifikasi_test")
tr["test"] = 0; te["test"] = 1; te["is_fraud"] = np.nan
A = pd.concat([tr, te], ignore_index=True); A["t"] = pd.to_datetime(A.waktu).astype("datetime64[ns]")
A = A.sort_values(["t", "id_transaksi"], kind="stable").reset_index(drop=True)
if QUICK: A = A[(A.t >= "2025-04-01")].reset_index(drop=True)
R = {}; D = pd.Timestamp
CUT = D("2025-08-31 23:59:59")

# ================= (a) AUDIT LABEL & FITUR =================
L = A[A.test == 0].copy(); L["wk"] = L.t.dt.to_period("W").dt.start_time
wk = L.groupby(["wk", "kanal"]).is_fraud.mean().unstack()
R["fraud_rate_bulanan"] = L.groupby(L.t.dt.to_period("M").astype(str)).is_fraud.mean().round(5).to_dict()
R["fraud_rate_mingguan_kanal_12minggu_terakhir"] = {str(k.date()): v for k, v in wk.tail(12).round(4).to_dict("index").items()}
# umur baris (minggu) saat penarikan data -> rasio laju fraud terhadap baseline matang (non-qris)
L["umur_mgg"] = ((CUT - L.t).dt.total_seconds() // (7 * 86400)).astype(int)
nq = L[L.kanal != "qris"]; base = nq[(nq.umur_mgg >= 6) & (nq.umur_mgg <= 14)].is_fraud.mean()
rec = {int(a): float(nq[nq.umur_mgg == a].is_fraud.mean() / base) for a in range(0, 6)}
R["rasio_pemulihan_label_per_umur_minggu_nonqris"] = rec
MAT = {"kartu": 28, "aplikasi": 28, "va": 28, "web": 21, "qris": 7}   # hari sampai label dianggap matang (dari kurva mingguan)
R["maturity_days"] = MAT
R["flag_review_manual"] = dict(fraud_rate_flag1=float(L[L.flag_review_manual == 1].is_fraud.mean()), fraud_rate_flag0=float(L[L.flag_review_manual == 0].is_fraud.mean()),
    share_fraud_with_flag=float(L[L.is_fraud == 1].flag_review_manual.mean()), mean_flag_train=float(L.flag_review_manual.mean()), mean_flag_test=float(A[A.test == 1].flag_review_manual.mean()))
fig, ax = plt.subplots(figsize=(7, 3.4)); wk.tail(16).plot(ax=ax, marker="o", ms=3); ax.set_ylabel("laju fraud"); ax.set_title("Laju fraud mingguan per kanal (16 minggu terakhir data latih)", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t2_label_matang.png"), dpi=150); plt.close()
# pergeseran pola qris
q = L[(L.kanal == "qris") & (L.t >= "2025-07-01")]; qa = q[q.t >= "2025-08-01"]; qb = q[q.t < "2025-08-01"]
cols = ["nominal", "jam", "umur_akun_hari", "umur_merchant_hari", "jarak_lokasi_km", "persen_saldo", "trx_1j"]
R["qris_shift"] = dict(rate_juli=float(qb.is_fraud.mean()), rate_agu=float(qa.is_fraud.mean()), n_fraud_agu=int(qa.is_fraud.sum()), n_fraud_juli=int(qb.is_fraud.sum()),
    fraud_agu_mean={c: float(qa[qa.is_fraud == 1][c].median()) for c in cols}, nonfraud_agu_mean={c: float(qa[qa.is_fraud == 0][c].median()) for c in cols},
    kategori_fraud_agu=qa[qa.is_fraud == 1].kategori_merchant.value_counts().head(3).to_dict())

# ================= (c) FITUR PERILAKU (hanya masa lalu) =================
A["ln"] = np.log(A.nominal)
gu = A.groupby("id_pengguna", sort=False)
A["u_n"] = gu.cumcount()
cs = gu.ln.cumsum() - A.ln; A["u_mean_ln"] = cs / A.u_n.replace(0, np.nan)
A["u_max_prior"] = A.groupby("id_pengguna", sort=False).ln.cummax().groupby(A.id_pengguna, sort=False).shift(1)
A["ln_vs_max"] = A.ln - A.u_max_prior; A["ln_vs_mean"] = A.ln - A.u_mean_ln
A["u_span_hari"] = (A.t - gu.t.transform("first")).dt.total_seconds() / 86400
A["pair_n"] = A.groupby(["id_pengguna", "id_merchant"], sort=False).cumcount()
A["u_kat_n"] = A.groupby(["id_pengguna", "kategori_merchant"], sort=False).cumcount()
A["u_kanal_n"] = A.groupby(["id_pengguna", "kanal"], sort=False).cumcount()
gm = A.groupby("id_merchant", sort=False); A["m_n"] = gm.cumcount()
A["m_newpair"] = (A.pair_n == 0).astype(int); gm = A.groupby("id_merchant", sort=False); A["m_newpair_share"] = (gm.m_newpair.cumsum() - A.m_newpair) / A.m_n.replace(0, np.nan)
gm = A.groupby("id_merchant", sort=False); A["m_ln_mean"] = (gm.ln.cumsum() - A.ln) / A.m_n.replace(0, np.nan); A["ln_vs_m"] = A.ln - A.m_ln_mean
A["dow"] = A.t.dt.dayofweek; A["dom"] = A.t.dt.day
A["log_saldo"] = np.log1p(A.saldo_sebelum.clip(lower=0)); A["log_nominal_24j"] = np.log1p(A.nominal_24j)
A["menit_na"] = A.menit_sejak_trx_terakhir.isna().astype(int)
for c in ["kanal", "tipe_perangkat", "kategori_merchant", "email_domain"]: A[c + "_c"] = A[c].astype("category")
A["y"] = A.is_fraud

def lagged_te(known_until, lag_days=28, m=50.0):
    """Laju fraud merchant dari label yang sudah matang (>= lag_days sebelum waktu baris & sebelum known_until)."""
    cap = (known_until - pd.Timedelta(days=lag_days)).value
    lab = A[(A.test == 0) & (A.t <= known_until - pd.Timedelta(days=lag_days))]
    prior = lab.y.mean(); out = np.full(len(A), prior); cnt = np.zeros(len(A))
    tq = np.minimum(A.t.values.astype("int64") - lag_days * 86400 * 10**9, cap)
    lm = {k: (v.t.values.astype("int64"), v.y.cumsum().values) for k, v in lab.groupby("id_merchant")}
    for mid, idx in A.groupby("id_merchant").indices.items():
        s = lm.get(mid)
        if s is None: continue
        pos = np.searchsorted(s[0], tq[idx], side="right"); f = np.where(pos > 0, s[1][np.maximum(pos - 1, 0)], 0.0)
        out[idx] = (f + m * prior) / (pos + m); cnt[idx] = pos
    return out, cnt

BASE = ["nominal", "jam", "hari_libur", "umur_akun_hari", "level_kyc", "umur_pengguna_tahun", "umur_perangkat_hari", "perangkat_baru",
        "jumlah_akun_per_perangkat", "ganti_password_7h", "gagal_login_24j", "jarak_lokasi_km", "negara_ip_beda", "kartu_tidak_hadir",
        "menit_sejak_trx_terakhir", "trx_1j", "trx_24j", "trx_7h", "nominal_24j", "rasio_nominal_rata2", "persen_saldo", "umur_merchant_hari",
        "refund_merchant_30h", "volume_merchant_30h", "kanal_c", "tipe_perangkat_c", "kategori_merchant_c", "email_domain_c",
        "dow", "dom", "log_saldo", "menit_na", "u_n", "u_mean_ln", "ln_vs_max", "ln_vs_mean", "u_span_hari", "pair_n", "u_kat_n", "u_kanal_n",
        "m_n", "m_newpair_share", "ln_vs_m"] + [f"v{i:02d}" for i in range(1, 51)]
def make_X(known_until, with_flag=False):
    X = A[BASE].copy(); te_, cnt = lagged_te(known_until); X["m_fraud_te"] = te_; X["m_lab_n"] = cnt
    if with_flag: X["flag_review_manual"] = A.flag_review_manual
    return X.astype({c: "float32" for c in X.columns if str(X[c].dtype) != "category"})

def mature_mask(known_until):
    age = (known_until - A.t).dt.total_seconds() / 86400
    return ((A.test == 0) & (A.t <= known_until) & (age >= A.kanal.map(MAT))).values

def clf(**kw):
    p = dict(learning_rate=0.1, max_iter=NIT, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED)
    p.update(kw); return HGBC(**p)
def fit(X, y, w=None, under=None):
    if under:
        r = np.random.RandomState(SEED); keep = (y == 1) | (r.rand(len(y)) < under); X, y = X[keep], y[keep]; w = None if w is None else w[keep]
    return clf().fit(X, y, sample_weight=w)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def platt(s, y): return LogisticRegression(C=1e6, max_iter=500).fit(logit(s).reshape(-1, 1), y)
def apply_platt(pl, s): return pl.predict_proba(logit(s).reshape(-1, 1))[:, 1]
def cost(y, nom, flag): return float(nom[(y == 1) & (flag == 0)].sum() + COST_REVIEW * flag.sum())
def best_c(y, nom, p, grid=np.geomspace(0.2, 6, 30)):
    cs = [(cost(y, nom, (p * nom > c * COST_REVIEW).astype(int)), c) for c in grid]; return min(cs)[1]
def metrics(y, nom, p, c=1.0):
    fl = (p * nom > c * COST_REVIEW).astype(int)
    return dict(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)), cost=cost(y, nom, fl), n_flag=int(fl.sum()),
                recall=float(((fl == 1) & (y == 1)).sum() / max(y.sum(), 1)), prec=float(((fl == 1) & (y == 1)).sum() / max(fl.sum(), 1)),
                cost_none=cost(y, nom, np.zeros(len(y), int)), cost_all=cost(y, nom, np.ones(len(y), int)), n=len(y), n_fraud=int(y.sum()))

# ================= (b,d) SKEMA VALIDASI & IMBALANCE =================
# Skema S: latih <= 31 Mei (matang) | kalibrasi + tuning ambang: Juni | evaluasi: Juli
K1 = D("2025-05-31 23:59:59"); X1 = make_X(K1)
trm = mature_mask(K1) & (A.t <= K1).values
cal = ((A.t > K1) & (A.t <= "2025-06-30 23:59:59") & (A.test == 0)).values; ev = ((A.t > "2025-06-30 23:59:59") & (A.t <= "2025-07-31 23:59:59") & (A.test == 0)).values
if QUICK: cal = ((A.t > K1) & (A.t <= "2025-06-30 23:59:59") & (A.test == 0)).values
yv = A.y.values; nom = A.nominal.values
def evaluate_variant(name, Xm, w=None, under=None, trmask=None, ytrain=None):
    m = fit(Xm[trmask if trmask is not None else trm], (ytrain if ytrain is not None else yv)[trmask if trmask is not None else trm].astype(int),
            None if w is None else w[trmask if trmask is not None else trm], under)
    sc, se = m.predict_proba(Xm[cal])[:, 1], m.predict_proba(Xm[ev])[:, 1]
    pl = platt(sc, yv[cal].astype(int)); pc, pe = apply_platt(pl, sc), apply_platt(pl, se); c = best_c(yv[cal], nom[cal], pc)
    r = metrics(yv[ev], nom[ev], pe, c); r["c_tuned"] = float(c); r["pr_auc_cal_window"] = float(average_precision_score(yv[cal], sc)); log(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, round(time.time() - t0)); return r, m, (pc, pe)
res = {}
res["HGB polos (tanpa penyeimbangan)"], mA, pa = evaluate_variant("plain", X1)
wpos = np.where(yv == 1, 20.0, 1.0)
res["Class weighting (bobot fraud 20x)"], _, _ = evaluate_variant("weight", X1, w=wpos)
res["Undersampling non-fraud (10%)"], _, _ = evaluate_variant("under", X1, under=0.10)
R["imbalance"] = res
# ablasi fitur terlarang dan label belum matang (dampak audit)
Xf = make_X(K1, with_flag=True); res_leak, _, _ = evaluate_variant("flag_review", Xf); R["leak_flag_review"] = dict(dengan_flag=res_leak, tanpa_flag=res["HGB polos (tanpa penyeimbangan)"]); del Xf; import gc; gc.collect()
rng = np.random.RandomState(SEED); age_w = ((K1 - A.t).dt.total_seconds() // (7 * 86400)).astype(int).values
y_naive = yv.copy(); inmat = (A.t <= K1).values & (A.test == 0).values
for a in range(0, 5):
    sel = inmat & (age_w == a) & (yv == 1); drop = rng.rand(len(A)) > rec.get(a, 1.0); y_naive[sel & drop] = 0
tr_all = inmat
r_naive, _, _ = evaluate_variant("label_naive", X1, trmask=tr_all, ytrain=y_naive)
R["label_maturity_impact"] = dict(label_belum_matang_dipakai=r_naive, label_matang_saja=res["HGB polos (tanpa penyeimbangan)"], n_fraud_hilang=int(((y_naive != yv) & inmat).sum()))
save_json(R, "t2_hasil_awal.json")

# ================= (e) PERGESERAN POLA: latih <= 30 Jun, kalibrasi Juli, uji 1-14 Agustus (baris matang) =================
K2 = D("2025-06-30 23:59:59"); X2 = make_X(K2); tr2 = mature_mask(K2) & (A.t <= K2).values
cal2 = ((A.t > K2) & (A.t <= "2025-07-31 23:59:59") & (A.test == 0)).values
age_now = (CUT - A.t).dt.total_seconds() / 86400
ev2 = ((A.t > "2025-07-31 23:59:59") & (A.t <= "2025-08-14 23:59:59") & (A.test == 0) & (age_now >= A.kanal.map(MAT))).values
ev2q = ev2 & (A.kanal == "qris").values
def halflife_w(mask, hl, ref):
    return 0.5 ** (((ref - A.t).dt.total_seconds() / 86400).clip(lower=0) / hl).values
shift = {}
for name, hl in [("tanpa bobot waktu", None), ("bobot recency (half-life 60 hari)", 60), ("bobot recency (half-life 30 hari)", 30)]:
    w = None if hl is None else halflife_w(tr2, hl, K2)
    m = fit(X2[tr2], yv[tr2].astype(int), None if w is None else w[tr2])
    if hl is None: m_old = m
    sc = m.predict_proba(X2[cal2])[:, 1]; pl = platt(sc, yv[cal2].astype(int)); pe = apply_platt(pl, m.predict_proba(X2[ev2])[:, 1])
    c = best_c(yv[cal2], nom[cal2], apply_platt(pl, sc)); r = metrics(yv[ev2], nom[ev2], pe, c)
    pq = pe[(A.kanal[ev2] == "qris").values]; r["pr_auc_qris"] = float(average_precision_score(yv[ev2q], pq)) if yv[ev2q].sum() > 0 else None; r["n_fraud_qris"] = int(yv[ev2q].sum()); shift[name] = r
    log("shift", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, round(time.time() - t0))
R["pattern_shift_eval"] = shift
# adaptasi: latih ulang bergulir sampai 7 Agustus (baris matang) lalu uji qris 8-24 Agustus (label qris matang cepat)
K2b = D("2025-08-07 23:59:59"); X2b = make_X(K2b); tr2b = mature_mask(K2b) & (A.t <= K2b).values
ev2b = ((A.t > K2b) & (A.t <= "2025-08-24 23:59:59") & (A.test == 0) & (A.kanal == "qris")).values
m_b = fit(X2b[tr2b], yv[tr2b].astype(int))
R["adaptasi_retrain"] = dict(n_fraud_qris_uji=int(yv[ev2b].sum()), n_uji=int(ev2b.sum()),
    pr_auc_qris_model_lama_30jun=float(average_precision_score(yv[ev2b], m_old.predict_proba(X2[ev2b])[:, 1])),
    pr_auc_qris_model_retrain_7agu=float(average_precision_score(yv[ev2b], m_b.predict_proba(X2b[ev2b])[:, 1])))
log("adaptasi", R["adaptasi_retrain"]); del X2b, m_b; import gc; gc.collect()
best_shift = max(shift, key=lambda k: shift[k]["pr_auc"]); HL = {"tanpa bobot waktu": None, "bobot recency (half-life 60 hari)": 60, "bobot recency (half-life 30 hari)": 30}[best_shift]; R["recency_dipilih"] = best_shift
save_json(R, "t2_hasil_awal.json")

# ================= (f) ANALISIS DI JULI (model plain skema S) =================
pc, pe = pa; Ev = A[ev].copy(); Ev["p"] = pe; Ev["tandai"] = (pe * Ev.nominal.values > R["imbalance"]["HGB polos (tanpa penyeimbangan)"]["c_tuned"] * COST_REVIEW).astype(int)
def seg(col):
    o = {}
    for k, d in Ev.groupby(col):
        y = d.y.values.astype(int)
        o[str(k)] = dict(n=len(d), n_fraud=int(y.sum()), rate=float(y.mean()), pr_auc=float(average_precision_score(y, d.p)) if y.sum() > 0 else None,
                         recall=float(((d.tandai == 1) & (d.y == 1)).sum() / max(y.sum(), 1)), n_flag=int(d.tandai.sum()),
                         cost=cost(y, d.nominal.values, d.tandai.values), cost_none=cost(y, d.nominal.values, np.zeros(len(d), int)))
    return o
R["segment"] = {c: seg(c) for c in ["kanal", "kategori_merchant"]}
# 20 fraud terbesar yang lolos (missed)
miss = Ev[(Ev.y == 1) & (Ev.tandai == 0)]; R["missed_summary"] = dict(n=len(miss), nominal_sum=float(miss.nominal.sum()), kanal=miss.kanal.value_counts().to_dict(), kategori=miss.kategori_merchant.value_counts().head(4).to_dict())
pat = {}
for c in ["perangkat_baru", "negara_ip_beda", "kartu_tidak_hadir", "ganti_password_7h", "hari_libur"]:
    pat[c] = L.groupby(c).is_fraud.mean().round(4).to_dict()
pat["jam_malam(0-5)"] = L.groupby(L.jam <= 5).is_fraud.mean().round(4).to_dict(); pat["email_domain"] = L.groupby("email_domain").is_fraud.mean().round(4).to_dict()
pat["jumlah_akun_per_perangkat>=3"] = L.groupby(L.jumlah_akun_per_perangkat >= 3).is_fraud.mean().round(4).to_dict()
pat["umur_perangkat_hari<=3"] = L.groupby(L.umur_perangkat_hari <= 3).is_fraud.mean().round(4).to_dict()
R["fraud_patterns"] = pat
sub = np.random.RandomState(SEED).choice(np.where(ev)[0], 15000, replace=False)
pi = permutation_importance(mA, X1.iloc[sub], yv[sub].astype(int), scoring=make_scorer(average_precision_score, response_method="predict_proba"), n_repeats=2, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X1.columns).sort_values(ascending=False); R["perm_importance"] = imp.head(15).round(5).to_dict()
fig, ax = plt.subplots(figsize=(6, 4.2)); imp.head(15)[::-1].plot.barh(ax=ax, color="#c63"); ax.set_title("Permutation importance (turunnya PR-AUC)", fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t2_importance.png"), dpi=150); plt.close()
# kurva biaya vs pengali ambang (Juli)
cs = np.geomspace(0.2, 6, 30); cc = [cost(Ev.y.values.astype(int), Ev.nominal.values, (Ev.p.values * Ev.nominal.values > c * COST_REVIEW).astype(int)) for c in cs]
fig, ax = plt.subplots(figsize=(5.5, 3.2)); ax.semilogx(cs, np.array(cc) / 1e6, marker="o", ms=3); ax.axhline(R["imbalance"]["HGB polos (tanpa penyeimbangan)"]["cost_none"] / 1e6, color="r", ls="--", label="flag none"); ax.set_xlabel("pengali ambang c (flag jika p x nominal > c x 25.000)"); ax.set_ylabel("biaya (juta Rp)"); ax.legend(fontsize=7); ax.set_title("Biaya vs ambang, validasi Juli", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t2_biaya.png"), dpi=150); plt.close()
save_json(R, "t2_hasil_awal.json")

# ================= MODEL FINAL =================
# kalibrasi Platt + pengali ambang diambil dari model latih<=30 Jun (recency terpilih), kalibrasi Juli;
# model final: semua baris latih matang (per kanal) sampai 31 Agu.
w2 = None if HL is None else halflife_w(tr2, HL, K2)
m2 = fit(X2[tr2], yv[tr2].astype(int), None if w2 is None else w2[tr2]); s2 = m2.predict_proba(X2[cal2])[:, 1]
PL = platt(s2, yv[cal2].astype(int)); C_STAR = best_c(yv[cal2], nom[cal2], apply_platt(PL, s2)); R["c_star"] = float(C_STAR)
R["platt"] = dict(coef=float(PL.coef_[0, 0]), intercept=float(PL.intercept_[0]))
K3 = CUT; X3 = make_X(K3); tr3 = mature_mask(K3)
w3 = None if HL is None else halflife_w(tr3, HL, K3)
m3 = fit(X3[tr3], yv[tr3].astype(int), None if w3 is None else w3[tr3])
tm = (A.test == 1).values; pt = apply_platt(PL, m3.predict_proba(X3[tm])[:, 1])
out = pd.DataFrame({"id_transaksi": A.id_transaksi[tm].values, "prob": np.clip(pt, 0, 1).round(6), "tandai": (pt * nom[tm] > C_STAR * COST_REVIEW).astype(int)})
out = out.set_index("id_transaksi").loc[te.id_transaksi].reset_index()
assert len(out) == len(te) and out.id_transaksi.is_unique and out.prob.between(0, 1).all() and set(out.tandai) <= {0, 1}
out.to_csv(os.path.join(ROOT, "prediksi_klasifikasi.csv"), index=False)
R["test_summary"] = dict(n=len(out), n_flag=int(out.tandai.sum()), pct_flag=float(out.tandai.mean()), mean_prob=float(out.prob.mean()), n_train_matang=int(tr3.sum()), n_fraud_train_matang=int(yv[tr3].sum()),
                          mean_prob_by_kanal=pd.Series(pt).groupby(A.kanal[tm].values).mean().round(5).to_dict())
save_json(R, "t2_hasil.json"); log("SELESAI T2", round(time.time() - t0))
