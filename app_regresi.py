import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.title("Regresi Linear")
st.write("Unggah file CSV yang memiliki kolom x dan y.")

file = st.file_uploader("Pilih file CSV", type=["csv"])

if file is None:
    st.stop()

try:
    data = pd.read_csv(file, sep=None, engine="python")
except (ValueError, pd.errors.ParserError):
    st.error("File CSV tidak bisa dibaca.")
    st.stop()

if not {"x", "y"}.issubset(data.columns):
    st.error("CSV harus memiliki kolom x dan y.")
    st.stop()

x = pd.to_numeric(data["x"], errors="coerce").to_numpy(dtype=float)
y = pd.to_numeric(data["y"], errors="coerce").to_numpy(dtype=float)

if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
    st.error("Masukkan minimal dua pasangan angka yang valid.")
    st.stop()

if np.unique(x).size < 2:
    st.error("Kolom x harus memiliki minimal dua nilai berbeda.")
    st.stop()

matriks = np.column_stack((np.ones(len(x)), x))
intersep, gradien = np.linalg.lstsq(matriks, y, rcond=None)[0]
prediksi = intersep + gradien * x
residual = y - prediksi

st.write(f"Persamaan regresi: ŷ = {intersep:.4f} {gradien:+.4f}x")
st.write(f"Jumlah residual: {residual.sum():.6e}")

hasil = pd.DataFrame({
    "x": x,
    "y": y,
    "Prediksi": prediksi,
    "Residual": residual,
})
st.dataframe(hasil)

urut = np.argsort(x)
fig, ax = plt.subplots()
ax.scatter(x, y, label="Data pengamatan")
ax.plot(x[urut], prediksi[urut], label="Garis regresi")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)
plt.close(fig)

x_baru = st.number_input("Masukkan nilai x untuk prediksi", value=float(x.max()))
st.write(f"Hasil prediksi: {intersep + gradien * x_baru:.4f}")