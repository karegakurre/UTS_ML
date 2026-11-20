"""Tugas 3 - Clustering segmentasi pelanggan. Jalankan: python t3_clustering.py"""
import os, time, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score as ARI, davies_bouldin_score, calinski_harabasz_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
t0 = time.time(); rs = np.random.RandomState(SEED); R = {}
d = load("clustering"); N = len(d)
POR = [c for c in d if c.startswith("porsi_") and "jam" not in c]; JAM = [c for c in d if c.startswith("porsi_jam")]
NOISE = [f"n{i:02d}" for i in range(1, 16)]

# ---------- (a) PREPROCESSING ----------
R["audit"] = dict(n=N, nan_rata2_keranjang=int(d.rata2_keranjang.isna().sum()), frekuensi0=int((d.frekuensi_90h == 0).sum()),
                  belanja0=int((d.belanja_90h == 0).sum()), frek30_gt_90=int((d.frekuensi_30h > d.frekuensi_90h).sum()),
                  belanja30_gt_90=int((d.belanja_30h > d.belanja_90h + 1).sum()), sum_porsi_min=float(d[POR].sum(1).min()), sum_porsi_max=float(d[POR].sum(1).max()),
                  sum_jam_min=float(d[JAM].sum(1).min()), sum_jam_max=float(d[JAM].sum(1).max()),
                  skew_belanja_90h=float(d.belanja_90h.skew()), skew_recency=float(d.recency_hari.skew()),
                  p999_belanja=float(d.belanja_90h.quantile(.999)), max_belanja=float(d.belanja_90h.max()), max_recency=float(d.recency_hari.max()),
                  max_sesi=float(d.sesi_per_minggu.max()), max_klik=float(d.klik_iklan_30h.max()))
def clr(df, cols):
    P = df[cols].clip(lower=0.005); L = np.log(P); return L.sub(L.mean(1), axis=0)
def build_X(d):
    X = pd.DataFrame(index=d.index)
    for c in ["recency_hari", "frekuensi_90h", "belanja_90h", "item_per_pesanan", "retur_90h", "ulasan_90h", "sesi_per_minggu", "wishlist", "klik_iklan_30h",
              "belanja_30h", "frekuensi_30h", "lama_member_bulan", "rasio_konversi", "jumlah_alamat"]: X[c] = np.log1p(d[c])
    X["rata2_keranjang"] = np.log1p(d.rata2_keranjang.fillna(0)); X["tidak_aktif"] = (d.frekuensi_90h == 0).astype(float)
    for df_, cols in [(d, POR), (d, JAM)]:
        C = clr(df_, cols); C.columns = ["clr_" + c for c in cols]; X = pd.concat([X, C], axis=1)
    for c in ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod", "rating_beri_rata2", "durasi_sesi_menit", "umur", "kota_tier"]: X[c] = d[c]
    for p in ["android", "ios", "web"]: X["dev_" + p] = (d.perangkat == p).astype(float)
    return X
X = build_X(d); lo, hi = X.quantile(.005), X.quantile(.995)
Xw = X.clip(lo, hi, axis=1)                                   # winsorize 0,5%-99,5% untuk pembentukan segmen
Z = pd.DataFrame(StandardScaler().fit_transform(Xw), columns=X.columns, index=X.index)

# ---------- (b) SELEKSI FITUR ----------
sidx = rs.choice(N, 30000, replace=False); Zs_full = pd.concat([Z, pd.DataFrame(StandardScaler().fit_transform(np.log1p(d[NOISE].clip(lower=0).abs())), columns=NOISE, index=d.index)], axis=1).iloc[sidx]
def eta2(Zs, lab):
    out = {}
    for c in Zs:
        g = Zs[c].groupby(lab); out[c] = float(((g.mean() - Zs[c].mean()) ** 2 * g.size()).sum() / ((Zs[c] - Zs[c].mean()) ** 2).sum())
    return pd.Series(out)
etas = []
for k in [4, 6, 8]:
    lab = KMeans(k, n_init=3, random_state=SEED).fit_predict(Zs_full.drop(columns=NOISE)); etas.append(eta2(Zs_full, lab))
ET = pd.concat(etas, axis=1); ET.columns = ["k4", "k6", "k8"]; ET["max"] = ET.max(1)
# noise: sertakan n01-n15 sebagai kontrol; ambang = 3 x eta2 maksimum fitur noise (min 0,02)
thr = max(0.02, 3 * ET.loc[NOISE, "max"].max()); KEEP = [c for c in X.columns if ET.loc[c, "max"] >= thr]
R["feature_selection"] = dict(threshold=float(thr), eta2=ET.round(4).to_dict("index"), dipertahankan=KEEP, dibuang=[c for c in X.columns if c not in KEEP], noise_eta2_max=float(ET.loc[NOISE, "max"].max()))
log("fitur dipertahankan", len(KEEP), "/", X.shape[1], "thr", thr, round(time.time() - t0))
ZK = Z[KEEP].values
pca = PCA(n_components=0.95, random_state=SEED).fit(ZK); P = pca.transform(ZK); R["pca"] = dict(n_comp=int(P.shape[1]), var=float(pca.explained_variance_ratio_.sum()))

# ---------- (d) JUMLAH CLUSTER: silhouette, BIC, stabilitas ----------
sub = rs.choice(N, 40000, replace=False); sil_i = rs.choice(40000, 8000, replace=False)
def stab(k, runs=4, frac=.5):
    labs = []
    for r in range(runs):
        ii = np.random.RandomState(100 + r).choice(N, int(N * frac), replace=False); km = KMeans(k, n_init=2, random_state=r).fit(ZK[ii]); labs.append(km.predict(ZK[sub]))
    return float(np.mean([ARI(labs[i], labs[j]) for i in range(runs) for j in range(i + 1, runs)]))
KS = list(range(2, 11)); sel = []
for k in KS:
    km = KMeans(k, n_init=3, random_state=SEED).fit(ZK[sub]); sil = silhouette_score(ZK[sub][sil_i], km.labels_[sil_i])
    gm = GaussianMixture(k, covariance_type="full", n_init=1, reg_covar=1e-4, random_state=SEED).fit(P[sub])
    sel.append(dict(k=k, inertia=float(km.inertia_), silhouette=float(sil), bic=float(gm.bic(P[sub])), stability_ari=stab(k)))
    log("k", sel[-1], round(time.time() - t0))
R["pilih_k"] = sel
S = pd.DataFrame(sel).set_index("k")
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].plot(S.index, S.silhouette, "o-"); ax[0].set_title("Silhouette (K-Means)", fontsize=9)
ax[1].plot(S.index, S.bic, "o-"); ax[1].set_title("BIC (GMM full)", fontsize=9)
ax[2].plot(S.index, S.stability_ari, "o-"); ax[2].set_title("Stabilitas ARI (subsampling 50%)", fontsize=9)
for a in ax: a.set_xlabel("k")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t3_pilih_k.png"), dpi=150); plt.close()
bic_d = S.bic.diff().fillna(0)
# k terpilih: terkecil k dengan stabilitas >= 0,85 dan silhouette dalam 90% maksimum, BIC terus turun tajam hanya sampai di situ
cand = S[(S.stability_ari >= 0.85) & (S.silhouette >= 0.9 * S.silhouette.max())]
K = int(os.environ.get("K", cand.index.max() if len(cand) else S.stability_ari.idxmax())); R["k_terpilih"] = K; log("K =", K)

# ---------- (c) PERBANDINGAN METODE ----------
def metr(lab, Zx):
    ii = np.random.RandomState(1).choice(len(Zx), min(8000, len(Zx)), replace=False)
    return dict(silhouette=float(silhouette_score(Zx[ii], lab[ii])), davies_bouldin=float(davies_bouldin_score(Zx, lab)), calinski_harabasz=float(calinski_harabasz_score(Zx, lab)))
res, labs = {}, {}
km = KMeans(K, n_init=5, random_state=SEED).fit(ZK); labs["K-Means"] = km.labels_
gmm = GaussianMixture(K, covariance_type="full", n_init=2, reg_covar=1e-4, random_state=SEED).fit(P); labs["Gaussian Mixture"] = gmm.predict(P)
ag_i = rs.choice(N, 20000, replace=False); ag = AgglomerativeClustering(K, linkage="ward").fit(P[ag_i])
knn = KNeighborsClassifier(15).fit(P[ag_i], ag.labels_); labs["Agglomerative (Ward, 20k + kNN)"] = knn.predict(P)
hd_i = rs.choice(N, 25000, replace=False); hd = HDBSCAN(min_cluster_size=400, min_samples=20).fit(P[hd_i])
hl = hd.labels_; R["hdbscan"] = dict(n_cluster=int(hl.max() + 1), noise_share=float((hl == -1).mean()))
if hl.max() >= 1:
    m_ = hl >= 0; knn2 = KNeighborsClassifier(15).fit(P[hd_i][m_], hl[m_]); labs["HDBSCAN (25k + kNN)"] = knn2.predict(P)
for nm, lab in labs.items():
    res[nm] = metr(lab, ZK); res[nm]["n_cluster"] = int(len(np.unique(lab))); res[nm]["ukuran_min_maks"] = [int(np.bincount(lab).min()), int(np.bincount(lab).max())]
names = list(labs); R["ari_antar_metode"] = {a: {b: float(ARI(labs[a], labs[b])) for b in names} for a in names}
# stabilitas tiap metode: 5 kali latih pada subsampel 50%, label seluruh data, ARI rerata antar-pasangan
def fit_pred(method, ii, seed):
    if method == "K-Means": return KMeans(K, n_init=2, random_state=seed).fit(ZK[ii]).predict(ZK)
    if method == "Gaussian Mixture": return GaussianMixture(K, covariance_type="full", n_init=1, reg_covar=1e-4, random_state=seed).fit(P[ii]).predict(P)
    if method.startswith("Agglo"):
        j = ii[:15000]; a = AgglomerativeClustering(K, linkage="ward").fit(P[j]); return KNeighborsClassifier(15).fit(P[j], a.labels_).predict(P)
    j = ii[:20000]; h = HDBSCAN(min_cluster_size=300, min_samples=20).fit(P[j]); m_ = h.labels_ >= 0
    return KNeighborsClassifier(15).fit(P[j][m_], h.labels_[m_]).predict(P) if h.labels_.max() >= 1 else np.zeros(N, int)
for nm in names:
    L_ = [fit_pred(nm, np.random.RandomState(200 + r).choice(N, N // 2, replace=False), r) for r in range(4)]
    sc = [ARI(L_[i], L_[j]) for i in range(4) for j in range(i + 1, 4)]; res[nm]["stabilitas_ari_mean"] = float(np.mean(sc)); res[nm]["stabilitas_ari_std"] = float(np.std(sc)); log(nm, res[nm], round(time.time() - t0))
R["metode"] = res
# metode final: stabilitas tertinggi di antara metode yang menghasilkan K cluster
elig = [n for n in names if res[n]["n_cluster"] == K] or names
FINAL = max(elig, key=lambda n: res[n]["stabilitas_ari_mean"] + res[n]["silhouette"]); R["metode_final"] = FINAL; lab = labs[FINAL].copy()

# ---------- (e) STABILITAS FINAL: bootstrap 10x, ARI antar-run ----------
BL = []
for r in range(10):
    ii = np.random.RandomState(300 + r).choice(N, N, replace=True); BL.append(fit_pred(FINAL, ii, 300 + r))
sc = [ARI(BL[i], BL[j]) for i in range(10) for j in range(i + 1, 10)]; R["stabilitas_bootstrap"] = dict(mean=float(np.mean(sc)), std=float(np.std(sc)), min=float(np.min(sc)), n_run=10, ari_vs_final=float(np.mean([ARI(b, lab) for b in BL])))
log("bootstrap", R["stabilitas_bootstrap"], round(time.time() - t0))

# ---------- (g) ANOMALI -1 ----------
ll = gmm.score_samples(P); med = np.median(ll); mad = np.median(np.abs(ll - med)) * 1.4826
share_rule = float((ll < med - 6 * mad).mean()); q = np.quantile(ll, [.005, .01, .02, .03, .05]); R["anomali"] = dict(med=float(med), mad=float(mad), share_rule_6mad=share_rule, quantiles_ll=q.tolist())
thr_ll = med - 6 * mad if 0.003 <= share_rule <= 0.04 else np.quantile(ll, 0.015)
anom = ll < thr_ll; R["anomali"]["share_final"] = float(anom.mean()); R["anomali"]["n"] = int(anom.sum())
fin = lab.copy(); fin[anom] = -1
out = pd.DataFrame({"id_pelanggan": d.id_pelanggan, "cluster": fin.astype(int)}); out.to_csv(os.path.join(ROOT, "cluster.csv"), index=False)
assert len(out) == N and out.id_pelanggan.is_unique and out.cluster.dtype.kind == "i"

# ---------- (f) PROFIL ----------
d2 = d.copy(); d2["cluster"] = fin; ok = d2[d2.cluster >= 0]
prof_cols = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod",
             "retur_90h", "ulasan_90h", "sesi_per_minggu", "rasio_konversi", "lama_member_bulan", "umur", "kota_tier", "belanja_30h", "klik_iklan_30h"] + POR + JAM
prof = d2.groupby("cluster")[prof_cols].median().round(3); prof["n"] = d2.groupby("cluster").size(); prof["pct"] = (prof.n / N * 100).round(2)
prof["pct_tidak_aktif"] = d2.groupby("cluster").frekuensi_90h.apply(lambda s: (s == 0).mean()).round(3)
for p in ["android", "ios", "web"]: prof["pct_" + p] = d2.groupby("cluster").perangkat.apply(lambda s: (s == p).mean()).round(3)
prof.to_csv(os.path.join(OUT_DIR, "t3_profil.csv"))
zs = ((ok.groupby("cluster")[prof_cols].mean() - ok[prof_cols].mean()) / ok[prof_cols].std()).round(2); zs.to_csv(os.path.join(OUT_DIR, "t3_profil_z.csv"))
R["top_fitur_per_cluster"] = {int(c): zs.loc[c].abs().sort_values(ascending=False).head(5).index.tolist() for c in zs.index}
R["ukuran_final"] = d2.cluster.value_counts().sort_index().to_dict()
# plot PCA 2D
fig, ax = plt.subplots(figsize=(5.5, 4.5)); ii = rs.choice(N, 15000, replace=False)
sc_ = ax.scatter(P[ii, 0], P[ii, 1], c=fin[ii], s=2, cmap="tab10"); ax.set_title(f"Segmen ({FINAL}, K={K}) pada 2 PC pertama; -1 = anomali", fontsize=9); plt.colorbar(sc_)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_t3_pca.png"), dpi=150); plt.close()
save_json(R, "t3_hasil.json"); log("SELESAI T3", round(time.time() - t0))
