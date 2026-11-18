import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([64, 65, 68, 73, 76, 77, 80, 85])

a = np.column_stack((np.ones(len(x)), x))
intersep, gradien = np.linalg.lstsq(a, y, rcond=None)[0]

x_baru = 9
prediksi = intersep + gradien * x_baru

print(f"Persamaan regresi: y = {intersep:.2f} {gradien:+.2f}x")
print(f"Prediksi untuk {x_baru} jam belajar = {prediksi:.2f}") 

y_prediksi = intersep + gradien * x
residual = y - y_prediksi
jumlah_residual = np.sum(residual)

print("Residual tiap data =", np.round(residual, 6))
print(f"Jumlah residual = {jumlah_residual:.6e}")