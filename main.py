import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & SCALER
# ===============================
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Prediksi CKD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# HEADER
# ===============================
st.markdown("""
<h1 style='text-align:center;'>🩺 Sistem Prediksi Penyakit Ginjal Kronis (CKD)</h1>
<p style='text-align:center;'>
Implementasi awal model <i>Machine Learning</i> menggunakan algoritma
<b>K-Nearest Neighbor (KNN)</b><br>

</p>
<hr>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("🧾 Input Data Pasien ")

# 1. DEMOGRAFI
st.sidebar.subheader("1️⃣ Data Demografi")
age = st.sidebar.number_input(
    "AGE – Age (Usia pasien dalam tahun)",
    1, 100, 45
)

# 2. TEKANAN DARAH & URIN
st.sidebar.subheader("2️⃣ Tekanan Darah & Urin")

bp = st.sidebar.number_input(
    "BP – Blood Pressure (Tekanan darah, mmHg)",
    50, 200, 80
)

sg = st.sidebar.number_input(
    "SG – Specific Gravity (Berat jenis urin)",
    1.000, 1.030, 1.020, step=0.005
)

al = st.sidebar.number_input(
    "AL – Albumin (Kadar albumin dalam urin, 0–5)",
    0, 5, 1
)

su = st.sidebar.number_input(
    "SU – Sugar (Kadar gula dalam urin, 0–5)",
    0, 5, 0
)

# 3. SEL DARAH & INFEKSI URIN
st.sidebar.subheader("3️⃣ Sel Darah & Infeksi Urin")

rbc = st.sidebar.number_input(
    "RBC – Red Blood Cells (Sel darah merah urin, 0=normal, 1=abnormal)",
    0, 1, 0
)

pc = st.sidebar.number_input(
    "PC – Pus Cell (Sel nanah urin, 0=normal, 1=abnormal)",
    0, 1, 0
)

pcc = st.sidebar.number_input(
    "PCC – Pus Cell Clumps (Gumpalan sel nanah, 0=tidak ada, 1=ada)",
    0, 1, 0
)

ba = st.sidebar.number_input(
    "BA – Bacteria (Bakteri dalam urin, 0=tidak ada, 1=ada)",
    0, 1, 0
)

# 4. PARAMETER DARAH
st.sidebar.subheader("4️⃣ Parameter Darah")

bgr = st.sidebar.number_input(
    "BGR – Blood Glucose Random (Gula darah acak, mg/dL)",
    50, 500, 120
)

bu = st.sidebar.number_input(
    "BU – Blood Urea (Kadar urea darah, mg/dL)",
    1, 200, 15
)

sc = st.sidebar.number_input(
    "SC – Serum Creatinine (Kreatinin serum, mg/dL)",
    0.1, 20.0, 1.2
)

sod = st.sidebar.number_input(
    "SOD – Sodium (Natrium darah, mEq/L)",
    100, 170, 140
)

pot = st.sidebar.number_input(
    "POT – Potassium (Kalium darah, mEq/L)",
    2.0, 7.0, 4.0
)

# 5. HEMATOLOGI
st.sidebar.subheader("5️⃣ Hematologi")

hemo = st.sidebar.number_input(
    "HEMO – Hemoglobin (g/dL)",
    3.0, 20.0, 14.0
)

pcv = st.sidebar.number_input(
    "PCV – Packed Cell Volume (Persentase volume sel darah merah)",
    10, 60, 45
)

wbcc = st.sidebar.number_input(
    "WBCC – White Blood Cell Count (Jumlah sel darah putih /mm³)",
    2000, 20000, 7000
)

rbcc = st.sidebar.number_input(
    "RBCC – Red Blood Cell Count (Jumlah sel darah merah, juta/µL)",
    2.0, 8.0, 5.0
)

# 6. PENYAKIT PENYERTA
st.sidebar.subheader("6️⃣ Penyakit Penyerta (0 / 1)")

htn = st.sidebar.number_input(
    "HTN – Hypertension (Hipertensi, 0=tidak, 1=ya)",
    0, 1, 0
)

dm = st.sidebar.number_input(
    "DM – Diabetes Mellitus (Diabetes, 0=tidak, 1=ya)",
    0, 1, 0
)

cad = st.sidebar.number_input(
    "CAD – Coronary Artery Disease (Penyakit jantung koroner, 0=tidak, 1=ya)",
    0, 1, 0
)

appet = st.sidebar.number_input(
    "APPET – Appetite (Nafsu makan pasien, 0=baik, 1=buruk)",
    0, 1, 0
)

pe = st.sidebar.number_input(
    "PE – Pedal Edema (Pembengkakan kaki, 0=tidak, 1=ya)",
    0, 1, 0
)

ane = st.sidebar.number_input(
    "ANE – Anemia (Kekurangan sel darah merah, 0=tidak, 1=ya)",
    0, 1, 0
)

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

st.subheader("📋 Data Input Pasien")
st.dataframe(input_data, use_container_width=True)

# ===============================
# PREDIKSI
# ===============================
st.markdown("---")
if st.button("🔍 Jalankan Prediksi"):
    input_scaled = scaler.transform(input_data)
    prediction = knn_model.predict(input_scaled)[0]

    st.subheader("📊 Hasil Prediksi Sistem")

    if prediction == 1:
        st.markdown("""
        <div style="
            padding:22px;
            background-color:#ffe6e6;
            border-left:6px solid #c62828;
            border-radius:10px;
        ">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:32px; color:#c62828;">⚠</span>
                <span style="
                    font-size:26px;
                    font-weight:700;
                    color:#000000;
                ">
                    Prediksi: CKD (Penyakit Ginjal Kronis)
                </span>
            </div>
            <p style="
                margin-top:10px;
                font-size:16px;
                color:#000000;
            ">
                Sistem memprediksi bahwa pasien berisiko mengalami Penyakit Ginjal Kronis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            padding:22px;
            background-color:#ddffdd;
            border-left:6px solid #2e7d32;
            border-radius:10px;
        ">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:32px; color:#2e7d32;">✔</span>
                <span style="
                    font-size:26px;
                    font-weight:700;
                    color:#000000;
                ">
                    Prediksi: Not CKD
                </span>
            </div>
            <p style="
                margin-top:10px;
                font-size:16px;
                color:#000000;
            ">
                Sistem memprediksi bahwa pasien tidak terdeteksi berisiko Penyakit Ginjal Kronis.
            </p>
        </div>
        """, unsafe_allow_html=True)



    st.info(
        "Catatan: Hasil prediksi bersifat simulasi komputasional "
        "dan tidak menggantikan diagnosis medis oleh tenaga kesehatan."
    )
