"""Aplikasi Streamlit — Deteksi Fraud Transaksi Kartu Kredit (DNN, model TFLite int8).

Jalankan dari root repo:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:                                    # LiteRT (ringan, tanpa TensorFlow penuh)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from preprocessing import preprocess_raw_rows  # noqa: E402

MODEL_DIR = ROOT / "outputs" / "models"
RESULT_DIR = ROOT / "outputs" / "results"
SAMPLE_CSV = ROOT / "data" / "contoh_transaksi_uji.csv"
RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_artifacts():
    # Fitur unik #2: model final terkompresi (TFLite, bobot int8) -> 10 KB
    model = Interpreter(model_path=str(MODEL_DIR / "fraud_dnn_int8dyn.tflite"))
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    thr = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    return model, scaler, thr


def predict(df_raw):
    model, scaler, thr = load_artifacts()
    X = preprocess_raw_rows(df_raw[RAW_COLS], scaler)
    inp, out_det = model.get_input_details()[0], model.get_output_details()[0]
    model.resize_tensor_input(inp["index"], [len(X), X.shape[1]])
    model.allocate_tensors()
    model.set_tensor(inp["index"], X)
    model.invoke()
    prob = model.get_tensor(out_det["index"]).ravel()
    out = df_raw.copy()
    out["prob_fraud"] = prob
    out["prediksi"] = np.where(prob >= thr, "FRAUD", "Normal")
    return out, thr


st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", page_icon="💳", layout="wide")
_, _, THR = load_artifacts()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Informasi model")
    st.write("**Arsitektur:** DNN 64-32-16 (BatchNorm, ReLU, Dropout)")
    tfl_path = MODEL_DIR / "fraud_dnn_int8dyn.tflite"
    st.metric("Ukuran model TFLite int8", f"{tfl_path.stat().st_size / 1024:.1f} KB")
    try:
        tfl = json.load(open(RESULT_DIR / "tflite_results.json"))
        st.caption(f"Model Keras asli: {tfl['keras']['size_kb']:.1f} KB · keputusan sama "
                   f"{100 * tfl['tflite_int8dyn']['pred_agreement']:.3f}% pada data uji")
    except Exception:
        pass
    st.metric("Threshold optimal (dari data validation)", f"{THR:.4f}")
    st.caption("Fitur unik 1: threshold dipilih dari data validation, bukan 0,5. "
               "Fitur unik 2: model dikompresi ke TensorFlow Lite int8.")

st.title("💳 Deteksi Fraud Transaksi Kartu Kredit")
st.caption("Deep Neural Network · dataset Credit Card Fraud Detection (ULB, Kaggle)")

tab1, tab2 = st.tabs(["Unggah CSV", "Input manual"])


def show_results(df):
    res, thr = predict(df)
    res["prediksi_thr_0.5"] = np.where(res["prob_fraud"] >= 0.5, "FRAUD", "Normal")
    n_f = int((res["prediksi"] == "FRAUD").sum())
    n_05 = int((res["prediksi_thr_0.5"] == "FRAUD").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah transaksi", f"{len(res):,}".replace(",", "."))
    c2.metric("Terdeteksi fraud (threshold optimal)", n_f, delta=f"{n_f - n_05:+d} vs thr 0,5")
    c3.metric("Threshold optimal", f"{thr:.4f}")
    cols = ["prob_fraud", "prediksi", "prediksi_thr_0.5", "Amount", "Time"]
    if "Class" in res.columns:
        res["label_asli"] = np.where(res["Class"] == 1, "FRAUD", "Normal")
        res["status"] = np.where(res["label_asli"] == res["prediksi"], "✅ Benar", "❌ Salah")
        jenis = {("FRAUD", "FRAUD"): "TP", ("Normal", "Normal"): "TN",
                 ("Normal", "FRAUD"): "FP", ("FRAUD", "Normal"): "FN"}
        res["jenis"] = [jenis[(a, p)] for a, p in zip(res["label_asli"], res["prediksi"])]
        tp = int((res.jenis == "TP").sum()); fp = int((res.jenis == "FP").sum())
        fn = int((res.jenis == "FN").sum())
        st.write(f"**Evaluasi terhadap label asli:** benar {int((res.status == '✅ Benar').sum())} "
                 f"dari {len(res)} · TP {tp} · FN {fn} · FP {fp} · "
                 f"recall {tp / max(tp + fn, 1):.2%} · precision {tp / max(tp + fp, 1):.2%}")
        cols = ["label_asli", "status", "jenis"] + cols
    with st.expander("Fitur unik 1 — perbandingan threshold 0,5 vs threshold optimal"):
        st.write(f"Dengan threshold 0,5 terdeteksi **{n_05}** transaksi fraud; dengan threshold "
                 f"optimal {thr:.4f} terdeteksi **{n_f}** transaksi fraud.")
    view = res[cols].sort_values("prob_fraud", ascending=False).head(500)
    st.dataframe(view, width="stretch", hide_index=True,
                 height=min(38 * (len(view) + 1) + 4, 760))
    st.download_button("Unduh hasil prediksi (CSV)", res.to_csv(index=False).encode(),
                       "hasil_prediksi.csv")


with tab1:
    st.write("Unggah file CSV berformat sama dengan dataset Kaggle (kolom `Time`, `V1`–`V28`, "
             "`Amount`; kolom `Class` opsional untuk mengecek prediksi benar/salah).")
    f = st.file_uploader("File CSV", type="csv")
    use_sample = SAMPLE_CSV.exists() and st.checkbox("Gunakan contoh data uji (20 transaksi)")
    df = pd.read_csv(f) if f is not None else (pd.read_csv(SAMPLE_CSV) if use_sample else None)
    if df is not None:
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"Kolom tidak ditemukan: {missing}")
        else:
            show_results(df)

with tab2:
    st.write("Isi nilai fitur satu transaksi (V1–V28 adalah komponen PCA), atau ambil dari "
             "contoh data uji.")
    sample = pd.read_csv(SAMPLE_CSV) if SAMPLE_CSV.exists() else None
    base = {c: (50.0 if c == "Amount" else 0.0) for c in RAW_COLS}
    true_label = None
    if sample is not None:
        opts = ["(kosong)"] + [f"Contoh {i + 1} — label asli: "
                               f"{'FRAUD' if r.Class == 1 else 'Normal'}"
                               for i, r in sample.iterrows()]
        pick = st.selectbox("Isi otomatis dari contoh", opts)
        if pick != "(kosong)":
            row = sample.iloc[opts.index(pick) - 1]
            base = {c: float(row[c]) for c in RAW_COLS}
            true_label = "FRAUD" if row["Class"] == 1 else "Normal"
    cols = st.columns(6)
    vals = {}
    for i, c in enumerate(RAW_COLS):
        vals[c] = cols[i % 6].number_input(c, value=base[c], format="%.4f",
                                           key=f"in_{c}_{base[c]}")
    if st.button("Prediksi", type="primary"):
        res, thr = predict(pd.DataFrame([vals]))
        p = float(res["prob_fraud"].iloc[0])
        label = res["prediksi"].iloc[0]
        (st.error if label == "FRAUD" else st.success)(
            f"Prediksi: **{label}** — P(fraud) = {p:.4f} (threshold {thr:.4f})")
        if true_label:
            st.info(f"Label asli: **{true_label}** → prediksi "
                    f"{'BENAR ✅' if true_label == label else 'SALAH ❌'}")
