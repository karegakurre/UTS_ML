"""Menyusun laporan.pdf dari hasil/*.json dan hasil/*.png (semua angka dibaca dari keluaran skrip)."""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from common import OUT_DIR, ROOT
J = lambda n: json.load(open(os.path.join(OUT_DIR, n)))
T1, T2, T3 = J("t1_hasil.json"), J("t2_hasil.json"), J("t3_hasil.json")
ss = getSampleStyleSheet(); B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceAfter=6); H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceAfter=4, spaceBefore=6)
S = []; P = lambda t, st=B: S.append(Paragraph(t, st)); H = lambda t: P(t, H1); H_ = lambda t: P(t, H2)
def tbl(rows, cw=None, fs=7.5):
    t = Table(rows, colWidths=cw, repeatRows=1); t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")])); S.append(t); S.append(Spacer(1, 6))
def fig(name, w=15):
    p = os.path.join(OUT_DIR, name)
    if os.path.exists(p):
        from PIL import Image as PI; im = PI.open(p); S.append(Image(p, width=w * cm, height=w * cm * im.size[1] / im.size[0])); S.append(Spacer(1, 4))
f = lambda x, n=2: f"{x:.{n}f}"
rp = lambda x: f"Rp {x/1e6:,.1f} jt"

P("<b>Laporan UTS Machine Learning - Hands-on ML: Regresi, Klasifikasi, dan Clustering</b>", ParagraphStyle("t", parent=H1, fontSize=16))
P("Seluruh angka dalam laporan dibangkitkan oleh kode di folder <i>kode/</i> (perintah tunggal: <i>bash kode/run_all.sh</i>). Seed 42. Pustaka: Python 3.12, scikit-learn 1.9.1, pandas, numpy, scipy, matplotlib, reportlab. Hanya scikit-learn yang dipakai untuk pemodelan (HistGradientBoosting, Ridge, KMeans, GaussianMixture, Agglomerative, HDBSCAN); SHAP tidak tersedia di lingkungan kami sehingga interpretasi memakai permutation importance dan partial dependence.")

# =================== TUGAS 1 ===================
H("Tugas 1 - Regresi: estimasi durasi pengiriman (P50 dan P90)")
a = T1["audit"]
H_("1a. Audit data")
P(f"<b>Fitur yang tidak tersedia saat pickup.</b> <i>jumlah_scan</i> bocor: rata-rata {f(a['scan_mean_train'])} di data latih tetapi {f(a['scan_mean_test'])} di data uji (scan terkumpul selama pengiriman; di data uji hanya scan saat pickup). Korelasinya dengan log-durasi {f(a['corr_scan_logy'])}. Fitur anonim <i>f04, f05, f22, f23, f17</i> berkorelasi dengan jumlah_scan ({', '.join(f'{k}: {f(v)}' for k, v in a['corr_proxy_with_scan'].items())}) sehingga diperlakukan sebagai proksi bocor dan dibuang. <i>f30</i> adalah tren linear terhadap waktu (rerata bulanan {', '.join(f'{v}' for v in a['f30_month_mean'].values())}) dengan korelasi ke target {f(a['f30_corr_y'],4)}: tidak informatif dan berbahaya untuk ekstrapolasi, dibuang.")
r = T1["random_vs_temporal"]
P(f"<b>Dampak.</b> Pada subsampel 120 ribu baris, pinball P50 dengan fitur bocor adalah {f(r['dengan_fitur_bocor']['random_cv_pin50'])} (CV acak) dan {f(r['dengan_fitur_bocor']['temporal_pin50'])} (temporal), sedangkan tanpa fitur bocor {f(r['tanpa_fitur_bocor']['random_cv_pin50'])} dan {f(r['tanpa_fitur_bocor']['temporal_pin50'])}. Jadi fitur bocor memberi kesan model jauh lebih baik daripada kenyataan saat prediksi (fitur itu tidak ada di uji).")
P(f"<b>Masalah kualitas data lain.</b> jarak_km bernilai 9999 ({a['jarak9999_train']} baris latih, {a['jarak9999_test']} uji) dan 0 ({a['jarak0_train']} baris) adalah sentinel (median durasinya sama dengan paket biasa), diubah menjadi NaN dengan penanda. berat_kg kosong {f(a['berat_na_train']*100,1)}% (data latih; sebagian besar layanan kargo) dan {f(a['berat_na_test']*100,1)}% (uji). Ada {a['n_dur_gt500']} paket dengan durasi lebih dari 500 jam (maks {a['dur_max']:.0f} jam). Model memakai target log dan loss kuantil sehingga tahan terhadap outlier; evaluasi tidak membuang outlier. Mitra kurir <b>{', '.join(a['mitra_baru_test'])}</b> hanya muncul di data uji ({a['n_mitra_baru_test']} baris): kategorinya diperlakukan sebagai tidak dikenal (NaN) agar model tidak mengandalkan identitas yang belum pernah dilihat.")
H_("1b. Validasi")
P("Skema temporal meniru penggunaan nyata (memprediksi bulan-bulan berikutnya): F1 = latih Jan-Apr, uji Mei-Jun; F2 = latih Jan-Jun, uji Jul-Agu. Tuning hanya memakai F1. Target encoding dihitung out-of-fold (5-fold) di dalam bagian latih. Pada pembanding acak 5-fold, hasil lebih optimistis (lihat angka di 1a).")
H_("1c. Model dan hasil validasi")
rows = [["Model", "F1 pin50", "F1 pin90", "F1 cov90", "F2 pin50", "F2 pin90", "F2 cov90"]]
t1, t2 = T1["temporal"]["F1: latih Jan-Apr -> uji Mei-Jun"], T1["temporal"]["F2: latih Jan-Jun -> uji Jul-Agu"]
for k in ["Baseline (median grup)", "Linear log-target (Ridge)", "HGB log-target + kuantil residual", "HGB quantile loss", "HGB quantile loss (tuned, tanpa fitur drift)", "HGB quantile loss (tuned + fitur drift)"]:
    x1, x2 = t1.get(k), t2.get(k); rows.append([k] + ([f(x1["pin50"]), f(x1["pin90"]), f(x1["cov90"])] if x1 else ["-"] * 3) + [f(x2["pin50"]), f(x2["pin90"]), f(x2["cov90"])])
tbl(rows, [6.2 * cm] + [1.8 * cm] * 6)
P(f"Baseline = median/kuantil per layanan x pulau tujuan. Linear = Ridge pada log(durasi) dengan one-hot dan interaksi layanan x log-jarak, kuantil dari residual per layanan x pulau. HGB kuantil dilatih terpisah untuk q=0,5 dan q=0,9 pada log-durasi (kuantil invarian terhadap transformasi monoton). Grid tuning di F1: max_leaf_nodes {{15,31,63}} x learning_rate {{0,05;0,1}} x min_samples_leaf {{100,400}}; terpilih {T1['best_params']}. Pinball dalam jam.")
H_("1d. Kategori kardinalitas tinggi dan kategori baru")
P("kota_tujuan (405 level), rute (kota_asal>kota_tujuan) dan mitra x layanan dikodekan dengan target encoding terlicinkan (m=30) yang dihitung out-of-fold; level tak dikenal mendapat rerata global. Kategori kecil (layanan, mitra, hub, kota_asal, pulau) memakai dukungan kategorikal HGB; kategori tak dikenal (mis. K13) diperlakukan sebagai nilai hilang. Selain itu ditambahkan fitur <i>mitra_drift</i>: rerata residual log-durasi mitra pada jendela 30 hari yang berakhir minimal 7 hari sebelum pickup (tidak melewati akhir data latih).")
H_("1e. Ketidakpastian dan kalibrasi")
c = T1["calibration"]["evaluasi_agustus"]
P("Cakupan P90 mentah di bawah 0,90, terutama karena mitra K07 berubah rezim sejak Juli (median durasi naik dari sekitar 60 ke 78 jam). Kami menguji tiga opsi pada Agustus (latih Jan-Jun, kalibrasi Juli): tanpa kalibrasi, conformal global (offset kuantil residual log 0,9), dan conformal per mitra (Mondrian, dengan shrinkage ke offset global).")
tbl([["Kalibrasi (uji Agustus)", "pin90", "cov90", "cov90 K07", "cov90 non-K07", "lebar rata-rata"]] + [[k, f(v["pin90"]), f(v["cov90"]), f(v["cov90_K07"]), f(v["cov90_non_K07"]), f(v["width"], 1)] for k, v in c.items()], [5 * cm, 2 * cm, 2 * cm, 2.4 * cm, 2.8 * cm, 2.6 * cm])
P("Conformal per mitra dipakai pada prediksi final: offset dihitung dari model yang dilatih Jan-Jul dan dikalibrasi pada Agustus, lalu diterapkan ke model akhir yang dilatih pada semua data; mitra tanpa data kalibrasi (K13) memakai offset global. Pasca-proses memastikan p90 >= p50.")
o = T1["outlier_impact"]
P(f"<b>Dampak outlier ekstrem</b> (uji Agustus): {o['n']} paket di atas 500 jam menaikkan pinball P90 dari {f(o['pin90_wo'])} menjadi {f(o['pin90_all'])} dan pinball P50 dari {f(o['pin50_wo'])} menjadi {f(o['pin50_all'])}; cakupan hampir tidak berubah ({f(o['cov90_wo'])} vs {f(o['cov90_all'])}), sehingga skor pinball pada uji tersembunyi akan banyak ditentukan oleh ekor ini.")
fig("fig_t1_cakupan.png", 15)
H_("1f. Analisis galat dan interpretasi")
eb = T1["error_by"]
rows = [["Segmen", "n", "pin50", "pin90", "cov90", "median aktual"]]
for g in ["layanan", "pulau_tujuan", "mitra_kurir"]:
    for k, v in eb[g].items(): rows.append([f"{g}={k}", v["n"], f(v["pin50"], 1), f(v["pin90"], 1), f(v["cov90"]), f(v["median_aktual"], 1)])
tbl(rows, [5 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.6 * cm], 7)
P("Galat absolut terbesar ada pada tujuan jauh (Papua, Maluku, Sulawesi, Kalimantan) dan layanan hemat; ini wajar karena durasinya lebih panjang dan lebih bervariasi. Cakupan P90 relatif rata antar segmen setelah kalibrasi per mitra; tanpa itu K07 jelas kurang tercakup.")
fig("fig_t1_importance.png", 9); fig("fig_t1_pdp.png", 15)
imp = list(T1["perm_importance"].items())[:6]
P("Permutation importance (kenaikan pinball q=0,5 pada skala log) menunjukkan pengaruh terbesar dari encoding mitra x layanan, jarak, rute, kota tujuan, dan antrean hub: " + ", ".join(f"{k} ({f(v,4)})" for k, v in imp) + ". Partial dependence P50 terhadap fitur teratas ditampilkan di atas; durasi naik dengan jarak dan dengan antrean hub asal. Fitur drift ikut menurunkan pinball P50 F2 sedikit (lihat tabel), tetapi manfaat terbesarnya baru terlihat bila model memiliki data pasca-Juli seperti pada model final.")
P(f"<b>Keterbatasan.</b> Sebaran prediksi uji: P50 median {T1['pred_summary']['p50']['50%']} jam, P90 median {T1['pred_summary']['p90']['50%']} jam. Perubahan rezim kurir lain di periode uji (Sep-Okt) tidak dapat diantisipasi; conformal per mitra hanya mengoreksi pergeseran yang sudah terlihat di Agustus.")

# =================== TUGAS 2 ===================
S.append(PageBreak()); H("Tugas 2 - Klasifikasi: deteksi fraud")
H_("2a. Audit label dan fitur")
rec = T2["rasio_pemulihan_label_per_umur_minggu_nonqris"]; mt = T2["maturity_days"]
P(f"<b>Pembentukan label.</b> Label fraud terkonfirmasi dengan jeda: laju fraud mingguan turun tajam di minggu-minggu terakhir (rasio terhadap baseline matang menurut umur baris saat penarikan data, non-qris: 0 minggu = {f(rec['0'])}; 1 = {f(rec['1'])}; 2 = {f(rec['2'])}; 3 = {f(rec['3'])}; 4 = {f(rec['4'])}). Berdasarkan kurva mingguan per kanal kami menetapkan umur minimal agar label dianggap matang: {mt}. Transaksi yang lebih muda dari batas itu <b>dikeluarkan dari data latih dan data evaluasi</b>.")
fig("fig_t2_label_matang.png", 12)
lm = T2["label_maturity_impact"]
P(f"<b>Dampak.</b> Dengan mensimulasikan label belum matang (mengikuti rasio pemulihan di atas; {lm['n_fraud_hilang']} label fraud dianggap 0), PR-AUC pada Juli turun dari {f(lm['label_matang_saja']['pr_auc'],3)} (hanya label matang) menjadi {f(lm['label_belum_matang_dipakai']['pr_auc'],3)}, dan biaya naik dari {rp(lm['label_matang_saja']['cost'])} menjadi {rp(lm['label_belum_matang_dipakai']['cost'])}.")
fl = T2["flag_review_manual"]; lk = T2["leak_flag_review"]
P(f"<b>flag_review_manual</b> adalah hasil peninjauan manual, bukan informasi saat transaksi: laju fraud {f(fl['fraud_rate_flag1']*100,1)}% bila flag=1 vs {f(fl['fraud_rate_flag0']*100,2)}% bila 0; {f(fl['share_fraud_with_flag']*100,0)}% fraud punya flag; rerata flag di uji ({f(fl['mean_flag_test']*100,2)}%) lebih rendah dari latih ({f(fl['mean_flag_train']*100,2)}%) karena peninjauan belum selesai. Bila dipakai, PR-AUC Juli melonjak ke {f(lk['dengan_flag']['pr_auc'],3)} (vs {f(lk['tanpa_flag']['pr_auc'],3)}) - angka semu. Fitur ini dibuang.")
H_("2b. Validasi berbasis waktu")
P("Skema S: latih sampai 31 Mei (hanya baris matang) -> kalibrasi probabilitas dan penyetelan ambang pada Juni -> evaluasi pada Juli (semua label sudah matang pada 31 Agustus). Skema pergeseran: latih sampai 30 Juni -> kalibrasi Juli -> uji 1-14 Agustus pada baris yang sudah matang per kanal. Fitur riwayat dihitung hanya dari transaksi sebelumnya.")
H_("2c. Fitur perilaku")
P("Dari seluruh transaksi (latih+uji) yang diurutkan waktu, fitur dihitung dari masa lalu saja: jumlah transaksi sebelumnya pengguna, rerata dan maksimum log-nominal sebelumnya (dan selisih transaksi ini terhadapnya), rentang hari sejak transaksi pertama, jumlah transaksi sebelumnya pada pasangan pengguna-merchant / kategori / kanal, volume merchant sebelumnya, proporsi pasangan baru pada merchant, dan laju fraud merchant yang dilicinkan dari label yang berumur minimal 28 hari dan tidak melewati akhir periode latih. Ditambah fitur asli (kecuali flag_review_manual) dan v01-v50.")
H_("2d. Ketidakseimbangan kelas")
rows = [["Pendekatan", "PR-AUC", "ROC-AUC", "n flag", "recall", "presisi", "biaya", "c"]]
for k, v in T2["imbalance"].items(): rows.append([k, f(v["pr_auc"], 4), f(v["roc_auc"], 4), v["n_flag"], f(v["recall"]), f(v["prec"]), rp(v["cost"]), f(v["c_tuned"])])
v0 = list(T2["imbalance"].values())[0]; rows.append(["Flag semua", "-", "-", v0["n"], "1.00", f(v0["n_fraud"] / v0["n"], 4), rp(v0["cost_all"]), "-"]); rows.append(["Flag tidak ada", "-", "-", 0, "0.00", "-", rp(v0["cost_none"]), "-"])
tbl(rows, [4.6 * cm, 1.6 * cm, 1.6 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 2.4 * cm, 1.0 * cm])
P("Setiap pendekatan dikalibrasi dengan Platt scaling pada Juni, lalu pengali ambang c (flag bila p x nominal > c x Rp 25.000) disetel pada Juni (tuning ambang). Perbedaan PR-AUC antar pendekatan sangat kecil (< 0,01) dan biaya berbeda sekitar 7% - tidak ada penyeimbangan yang jelas lebih baik, sehingga model final memakai HGB tanpa bobot kelas (probabilitas lebih alami terkalibrasi) ditambah penyetelan ambang berbasis biaya.")
H_("2e. Pergeseran pola")
q = T2["qris_shift"]
P(f"Laju fraud kanal qris naik dari {f(q['rate_juli']*100,2)}% (Juli, {q['n_fraud_juli']} kasus) menjadi {f(q['rate_agu']*100,2)}% (Agustus, {q['n_fraud_agu']} kasus), terkonsentrasi di kategori {q['kategori_fraud_agu']}. Fraud qris Agustus memiliki median umur merchant {f(q['fraud_agu_mean']['umur_merchant_hari'],0)} hari (vs {f(q['nonfraud_agu_mean']['umur_merchant_hari'],0)} hari non-fraud) dan nominal median {rp(q['fraud_agu_mean']['nominal'])} (vs {rp(q['nonfraud_agu_mean']['nominal'])}): merchant baru dengan nominal besar.")
rows = [["Skema (uji 1-14 Agu)", "PR-AUC", "PR-AUC qris", "recall", "biaya"]] + [[k, f(v["pr_auc"], 3), f(v["pr_auc_qris"], 3), f(v["recall"]), rp(v["cost"])] for k, v in T2["pattern_shift_eval"].items()]
tbl(rows, [6 * cm, 2 * cm, 2.5 * cm, 2 * cm, 3 * cm])
ad = T2["adaptasi_retrain"]
P(f"PR-AUC turun dari sekitar 0,76 (Juli) ke sekitar 0,40-0,45 pada awal Agustus: model memang terpukul pergeseran. Pembobotan recency (half-life 60 hari) memperbaiki PR-AUC dan dipilih. Pelatihan ulang bergulir sampai 7 Agustus <b>tidak</b> memperbaiki deteksi qris (PR-AUC qris {f(ad['pr_auc_qris_model_retrain_7agu'],3)} vs {f(ad['pr_auc_qris_model_lama_30jun'],3)}; hanya {ad['n_fraud_qris_uji']} kasus uji), jadi kami tidak mengklaim adaptasi berhasil: kasus qris baru terlalu sedikit untuk dipelajari model. Model final dilatih pada seluruh label matang sampai 31 Agustus sehingga mencakup kasus Agustus yang sudah terkonfirmasi.")
H_("2f. Kalibrasi dan keputusan berbasis biaya")
P(f"Probabilitas dikalibrasi dengan Platt scaling (koefisien {f(T2['platt']['coef'],3)}, intersep {f(T2['platt']['intercept'],3)}) dari model Jan-Jun pada Juli. Aturan keputusan: tandai bila p x nominal > c* x Rp 25.000 dengan c* = {f(T2['c_star'])} (disetel pada Juli). Rasionalnya: fraud yang ditandai tertahan tanpa kerugian, sehingga menandai bernilai bila peluang x nominal melebihi biaya tinjau. Pada Juli (skema S) biaya 'flag semua' {rp(v0['cost_all'])}, 'flag tidak ada' {rp(v0['cost_none'])}, aturan berbasis biaya sekitar {rp(list(T2['imbalance'].values())[0]['cost'])}.")
fig("fig_t2_biaya.png", 9)
H_("2g. Analisis galat dan pola fraud")
rows = [["Kanal", "n", "fraud", "laju", "PR-AUC", "recall", "biaya", "biaya flag-none"]]
for k, v in T2["segment"]["kanal"].items(): rows.append([k, v["n"], v["n_fraud"], f(v["rate"] * 100, 2) + "%", f(v["pr_auc"], 3) if v["pr_auc"] is not None else "-", f(v["recall"]), rp(v["cost"]), rp(v["cost_none"])])
tbl(rows, [2 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 2.6 * cm, 3 * cm])
ms = T2["missed_summary"]; fp = T2["fraud_patterns"]
P(f"Fraud yang lolos pada Juli: {ms['n']} kasus bernilai total {rp(ms['nominal_sum'])}, terbanyak di {ms['kanal']} dan kategori {ms['kategori']}. Kanal qris paling sulit (PR-AUC rendah karena hanya 10 kasus di Juli). Pola fraud yang konsisten: perangkat baru (laju {f(fp['perangkat_baru']['1']*100,1)}% vs {f(fp['perangkat_baru']['0']*100,2)}%), IP negara berbeda ({f(fp['negara_ip_beda']['1']*100,1)}% vs {f(fp['negara_ip_beda']['0']*100,2)}%), ganti kata sandi 7 hari terakhir ({f(fp['ganti_password_7h']['1']*100,1)}% vs {f(fp['ganti_password_7h']['0']*100,2)}%), jam 00-05 ({f(fp['jam_malam(0-5)']['true']*100,1)}% vs {f(fp['jam_malam(0-5)']['false']*100,2)}%), kartu tanpa kehadiran ({f(fp['kartu_tidak_hadir']['1']*100,1)}% vs {f(fp['kartu_tidak_hadir']['0']*100,2)}%), email sekali pakai ({f(fp['email_domain']['sekali_pakai']*100,1)}%).")
fig("fig_t2_importance.png", 9)
ts = T2["test_summary"]
P(f"<b>Hasil uji.</b> Model final dilatih pada {ts['n_train_matang']:,} transaksi matang ({ts['n_fraud_train_matang']} fraud). Pada {ts['n']:,} transaksi uji, {ts['n_flag']:,} ({f(ts['pct_flag']*100,2)}%) ditandai; rerata probabilitas {f(ts['mean_prob']*100,2)}%. Catatan: fitur m_n (volume merchant sebelumnya) dan jumlah transaksi pengguna bergantung pada urutan waktu seluruh data, wajar di produksi tetapi perlu pemantauan.")

# =================== TUGAS 3 ===================
S.append(PageBreak()); H("Tugas 3 - Clustering: segmentasi pelanggan")
a = T3["audit"]
H_("3a. Preprocessing")
P(f"{a['n']:,} pelanggan. Variabel moneter dan hitungan sangat miring (skewness belanja_90h {f(a['skew_belanja_90h'],1)}; maks belanja {a['max_belanja']:.2e} vs persentil 99,9 {a['p999_belanja']:.2e}) sehingga ditransformasi log1p. Porsi kategori (16) dan porsi jam (4) bersifat <b>kompositional</b> (jumlah 1): kami memakai transformasi centered log-ratio (nilai nol diganti 0,005) agar jarak Euclid bermakna. rata2_keranjang kosong untuk {a['nan_rata2_keranjang']:,} pelanggan yang tidak pernah belanja (frekuensi_90h=0; sama persis): diisi 0 dan ditambah penanda <i>tidak_aktif</i>. kota_tier diperlakukan ordinal, perangkat di-one-hot. Outlier ditangani dengan winsorizing persentil 0,5-99,5 sebelum standardisasi; pelanggan ekstrem tetap diperiksa pada langkah anomali. Ditemukan juga {a['belanja30_gt_90']} baris dengan belanja_30h sedikit melebihi belanja_90h (kemungkinan pembulatan/penarikan), dibiarkan.")
H_("3b. Seleksi fitur")
fs = T3["feature_selection"]
P(f"Kami menaruh 15 kolom CRM anonim n01-n15 sebagai kontrol. Untuk setiap fitur dihitung eta-kuadrat (proporsi varians yang dijelaskan oleh label cluster K-Means) pada k=4, 6, 8 (sampel 30 ribu); nilai maksimum eta-kuadrat n01-n15 hanya {f(fs['noise_eta2_max'],4)}, artinya <b>seluruh n01-n15 noise</b>. Ambang pemilihan = maks(0,02; 3 x eta-kuadrat noise maksimum) = {f(fs['threshold'],3)}. Dipertahankan {len(fs['dipertahankan'])} dari 47 fitur; dibuang: {', '.join(fs['dibuang'])}. Keterbatasan: eta-kuadrat bersifat agak sirkular karena label berasal dari fitur itu sendiri; kontrol noise yang menjadi bukti utamanya. Data kemudian direduksi PCA ({T3['pca']['n_comp']} komponen, {f(T3['pca']['var']*100,1)}% varians) hanya untuk GMM/Agglomerative/HDBSCAN; K-Means memakai fitur terstandardisasi penuh.")
H_("3c-3d. Metode dan jumlah cluster")
S_ = T3["pilih_k"]
tbl([["k", "silhouette (KM)", "BIC (GMM)", "stabilitas ARI (KM, 50%)"]] + [[x["k"], f(x["silhouette"], 3), f"{x['bic']:.0f}", f(x["stability_ari"], 3)] for x in S_], [1.2 * cm, 3.6 * cm, 3.6 * cm, 5 * cm])
fig("fig_t3_pilih_k.png", 15)
P(f"Kriteria: silhouette tertinggi pada k=3 ({f(S_[1]['silhouette'],3)}) lalu menurun landai; BIC GMM tidak monoton (GMM full-covariance tidak stabil) sehingga tidak dijadikan penentu; stabilitas ARI sangat tinggi untuk k=3-5 (0,97-0,99) dan turun tajam pada k>=6. Struktur segmen tergolong lemah (silhouette sekitar 0,1), sehingga stabilitas menjadi kriteria utama: dipilih k = {T3['k_terpilih']}, k terbesar yang masih stabil (ARI >= 0,85) dengan silhouette dalam 90% dari maksimum, agar segmen cukup rinci untuk aksi pemasaran.")
rows = [["Metode", "n cluster", "silhouette", "Davies-Bouldin", "Calinski-Harabasz", "stabilitas ARI (mean+-sd)"]]
for k, v in T3["metode"].items(): rows.append([k, v["n_cluster"], f(v["silhouette"], 3), f(v["davies_bouldin"]), f"{v['calinski_harabasz']:.0f}", f"{f(v['stabilitas_ari_mean'],3)} +- {f(v['stabilitas_ari_std'],3)}"])
tbl(rows, [5 * cm, 1.6 * cm, 1.8 * cm, 2.4 * cm, 2.6 * cm, 3.6 * cm])
P(f"Empat metode dibandingkan: K-Means, Gaussian Mixture (full covariance), Agglomerative Ward (pada 20 ribu sampel, sisanya diberi label dengan kNN), dan HDBSCAN (pada 25 ribu sampel + kNN). HDBSCAN hanya menemukan {T3['hdbscan']['n_cluster']} cluster (memisahkan pelanggan tidak aktif dari semua yang lain; {f(T3['hdbscan']['noise_share']*100,1)}% noise) sehingga silhouette-nya tinggi tetapi tidak berguna untuk segmentasi; ia tidak memenuhi K={T3['k_terpilih']}. GMM sangat tidak stabil. Metode final: <b>{T3['metode_final']}</b> (stabilitas tertinggi di antara metode dengan K cluster).")
H_("3e. Stabilitas")
sb = T3["stabilitas_bootstrap"]
P(f"Bootstrap {sb['n_run']} kali (resampling dengan pengembalian, K-Means dilatih ulang, seluruh pelanggan diberi label): ARI antar-run rerata {f(sb['mean'],3)} (sd {f(sb['std'],3)}, minimum {f(sb['min'],3)}); ARI terhadap segmentasi final rerata {f(sb['ari_vs_final'],3)}. Dengan subsampling 50% empat kali, ARI antar-run {f(T3['metode']['K-Means']['stabilitas_ari_mean'],3)}. Ada variasi antar-run (sebagian run memecah dua segmen besar secara berbeda), tetapi pelanggan tidak aktif dan segmen keranjang besar selalu terbentuk.")
H_("3f. Profil segmen dan aksi")
import pandas as pd
pr = pd.read_csv(os.path.join(OUT_DIR, "t3_profil.csv"), index_col=0)
NAMES = {0: ("Mulai pasif", "Aktivitas rendah dan lama tidak belanja; COD tinggi.", "Kampanye win-back: voucher kembali belanja berbatas waktu + pengingat; dorong pembayaran non-COD."),
         1: ("Setia frekuen", "Sering belanja, belanja besar total, recency rendah; dominan makanan-minuman, pagi hari.", "Program loyalitas / langganan, bundling kebutuhan harian, pengiriman prioritas pagi."),
         2: ("Pemburu promo muda", "Sangat tergantung voucher dan flash sale, keranjang kecil, usia muda, fashion dan malam hari.", "Flash sale malam hari kategori fashion/kecantikan dengan syarat minimum belanja untuk menaikkan nilai keranjang."),
         3: ("Premium keranjang besar", "Keranjang besar, jarang memakai promo, porsi elektronik tinggi, usia lebih tua.", "Cross-sell elektronik/gadget, garansi dan cicilan paylater; hindari diskon besar."),
         4: ("Tidak aktif 90 hari", "Tidak ada transaksi 90 hari terakhir.", "Reaktivasi: survei singkat + penawaran selamat datang kembali; hentikan iklan berbayar bila tidak merespons."),
         -1: ("Anomali", "Tidak cocok dengan segmen mana pun (kepadatan GMM terendah).", "Tinjau manual (kemungkinan bot / data tidak wajar) sebelum dimasukkan ke kampanye.")}
rows = [["Cluster", "Nama", "n (%)", "recency", "frek.", "belanja 90h", "keranjang", "voucher", "flash", "COD", "usia"]]
for c in [0, 1, 2, 3, 4, -1]:
    r_ = pr.loc[c]; rows.append([c, NAMES[c][0], f"{int(r_.n):,} ({r_.pct}%)", f(r_.recency_hari, 1), f(r_.frekuensi_90h, 0), f"{r_.belanja_90h/1e6:.2f} jt", f"{r_.rata2_keranjang/1e3:.0f} rb" if r_.rata2_keranjang == r_.rata2_keranjang else "-", f(r_.persen_voucher), f(r_.persen_flash_sale), f(r_.persen_cod), f(r_.umur, 0)])
tbl(rows, [1.1 * cm, 3 * cm, 2.4 * cm, 1.3 * cm, 1 * cm, 1.8 * cm, 1.6 * cm, 1.2 * cm, 1.1 * cm, 1.1 * cm, 1 * cm], 7)
rows = [["Cluster", "Profil", "Aksi pemasaran"]] + [[c, Paragraph(NAMES[c][1], B), Paragraph(NAMES[c][2], B)] for c in [0, 1, 2, 3, 4, -1]]
tbl(rows, [1.4 * cm, 7.3 * cm, 8.4 * cm], 8)
P("Angka dalam tabel adalah median segmen (data selengkapnya di hasil/t3_profil.csv dan t3_profil_z.csv). Nama dan aksi disusun dari profil tersebut dan merupakan interpretasi kami, bukan label dari data.")
H_("3g. Anomali (-1)")
an = T3["anomali"]
P(f"Anomali ditentukan dari log-kepadatan GMM (pada ruang PCA). Aturan median - 6 x MAD hanya menandai {f(an['share_rule_6mad']*100,3)}% pelanggan, terlalu sedikit; kami memakai kuantil 1,5% terendah sehingga {an['n']} pelanggan ({f(an['share_final']*100,1)}%) diberi label -1. Persentase ini dipilih secara konservatif karena anomali referensi dikeluarkan dari skor ARI sedangkan pelanggan normal yang salah diberi -1 akan merugikan; pemilihan 1,5% adalah penilaian kami, tanpa validasi terhadap referensi.")
fig("fig_t3_pca.png", 9)

# =================== PENUTUP ===================
H("Reproduksi, AI, dan keterbatasan")
P("<b>Reproduksi.</b> Letakkan file *.csv.gz dari data_uts_ml.zip di kode/data/ lalu jalankan <i>bash kode/run_all.sh</i> (+-25 menit pada 1 CPU, RAM 4 GB). Skrip menulis tiga file prediksi, angka (hasil/*.json) dan gambar, lalu laporan ini.")
P("<b>Penggunaan AI.</b> Kode dan draf laporan ini dihasilkan dengan bantuan Claude (Anthropic) berdasarkan soal UTS dan data. Anggota kelompok wajib memverifikasi, mereproduksi, dan memahami seluruh isinya untuk ujian lisan; rincian verifikasi dituliskan di kontribusi.md.")
P("<b>Keterbatasan yang perlu dicatat.</b> (1) Tidak ada label uji sehingga performa uji tersembunyi tidak diketahui; semua angka adalah validasi temporal. (2) Kalibrasi interval regresi memakai offset dari Agustus; perubahan rezim baru di Sep-Okt tidak tertangkap. (3) Deteksi fraud qris baru lemah. (4) Struktur cluster lemah (silhouette sekitar 0,1); kecocokan dengan segmen referensi tidak dapat diverifikasi. (5) Tidak ada LightGBM/XGBoost/SHAP di lingkungan kami; digunakan scikit-learn.")
SimpleDocTemplate(os.path.join(ROOT, "laporan.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Laporan UTS Machine Learning").build(S)
print("laporan.pdf dibuat")
