import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL & SCALER
# ===============================
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prediksi CKD", layout="wide")

st.title("🩺 Sistem Prediksi Penyakit Ginjal Kronis (CKD)")
st.markdown("""
Aplikasi ini ditujukan untuk **pengguna awam**.  
Input disederhanakan menjadi **Normal / Tidak Normal** berdasarkan standar medis.  
⚠️ *Hasil prediksi tidak menggantikan diagnosis dokter.*
""")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("🧾 Input Data Pasien")

age = st.sidebar.number_input("Usia (tahun)", 1, 100, 45)

hipertensi = st.sidebar.selectbox("Hipertensi", ["Tidak", "Ya"])
bp = 140 if hipertensi == "Ya" else 120

sg_status = st.sidebar.selectbox("Berat Jenis Urin", ["Normal", "Tidak Normal"])
sg = 1.020 if sg_status == "Normal" else 1.005

albumin = st.sidebar.selectbox("Albumin Urin", ["Normal", "Tidak Normal"])
al = 0 if albumin == "Normal" else 2

sugar = st.sidebar.selectbox("Gula dalam Urin", ["Tidak", "Ada"])
su = 0 if sugar == "Tidak" else 2

rbc = st.sidebar.selectbox("Sel Darah Merah (Urin)", ["Normal", "Abnormal"])
rbc = 0 if rbc == "Normal" else 1

pc = st.sidebar.selectbox("Pus Cell", ["Normal", "Abnormal"])
pc = 0 if pc == "Normal" else 1

pcc = st.sidebar.selectbox("Pus Cell Clumps", ["Tidak Ada", "Ada"])
pcc = 0 if pcc == "Tidak Ada" else 1

ba = st.sidebar.selectbox("Bakteri dalam Urin", ["Tidak Ada", "Ada"])
ba = 0 if ba == "Tidak Ada" else 1

bgr_status = st.sidebar.selectbox("Gula Darah Acak", ["Normal", "Tinggi"])
bgr = 120 if bgr_status == "Normal" else 180

bu_status = st.sidebar.selectbox("Urea Darah", ["Normal", "Tinggi"])
bu = 15 if bu_status == "Normal" else 40

sc_status = st.sidebar.selectbox("Kreatinin Serum", ["Normal", "Tinggi"])
sc = 1.0 if sc_status == "Normal" else 3.0

sod_status = st.sidebar.selectbox("Natrium", ["Normal", "Tidak Normal"])
sod = 140 if sod_status == "Normal" else 130

pot_status = st.sidebar.selectbox("Kalium", ["Normal", "Tidak Normal"])
pot = 4.0 if pot_status == "Normal" else 6.0

hemo_status = st.sidebar.selectbox("Hemoglobin", ["Normal", "Rendah"])
hemo = 14 if hemo_status == "Normal" else 9

pcv_status = st.sidebar.selectbox("Packed Cell Volume", ["Normal", "Rendah"])
pcv = 45 if pcv_status == "Normal" else 30

wbcc_status = st.sidebar.selectbox("Sel Darah Putih", ["Normal", "Tidak Normal"])
wbcc = 7000 if wbcc_status == "Normal" else 13000

rbcc_status = st.sidebar.selectbox("Jumlah Sel Darah Merah", ["Normal", "Rendah"])
rbcc = 5.0 if rbcc_status == "Normal" else 3.5

htn = 1 if hipertensi == "Ya" else 0
dm = st.sidebar.selectbox("Diabetes", ["Tidak", "Ya"])
dm = 1 if dm == "Ya" else 0

cad = st.sidebar.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
cad = 1 if cad == "Ya" else 0

appet = st.sidebar.selectbox("Nafsu Makan", ["Baik", "Buruk"])
appet = 0 if appet == "Baik" else 1

pe = st.sidebar.selectbox("Pembengkakan Kaki", ["Tidak", "Ya"])
pe = 1 if pe == "Ya" else 0

ane = st.sidebar.selectbox("Anemia", ["Tidak", "Ya"])
ane = 1 if ane == "Ya" else 0

# ===============================
# DATAFRAME INPUT
# ===============================
input_data = pd.DataFrame([[
    age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
    bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
    htn, dm, cad, appet, pe, ane
]], columns=[
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr',
    'bu','sc','sod','pot','hemo','pcv','wbcc','rbcc',
    'htn','dm','cad','appet','pe','ane'
])


# ===============================
# PREDIKSI
# ===============================
if st.button("🔍 Prediksi CKD"):
    input_scaled = scaler.transform(input_data)
    prediction = knn_model.predict(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ TERDETEKSI RISIKO CKD")
    else:
        st.success("✅ TIDAK TERDETEKSI CKD")
