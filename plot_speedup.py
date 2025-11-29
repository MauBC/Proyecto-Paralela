import numpy as np
import matplotlib.pyplot as plt

# Cargar CSV
data = np.loadtxt("knn_times_v1.csv", delimiter=",")

# Extraer columnas
p = data[:,0].astype(int)
t_total = data[:,2]

# Ordenar por p
idx = np.argsort(p)
p = p[idx]
t_total = t_total[idx]

# Encontrar T(1)
if 1 in p:
    t1 = t_total[p == 1][0]
else:
    raise ValueError("❌ ERROR: No existe medición con p = 1 en el CSV.")

# Calcular Speedup y Eficiencia
speedup = t1 / t_total
efficiency = speedup / p

# --- Graficar ---
plt.figure(figsize=(10,5))

# Speedup
plt.subplot(1,2,1)
plt.plot(p, speedup, marker="o")
plt.title("Speedup vs Procesos")
plt.xlabel("Procesos (p)")
plt.ylabel("Speedup (S)")
plt.grid(True)

# Eficiencia
plt.subplot(1,2,2)
plt.plot(p, efficiency, marker="o")
plt.title("Eficiencia vs Procesos")
plt.xlabel("Procesos (p)")
plt.ylabel("Eficiencia (S/p)")
plt.grid(True)

plt.tight_layout()
#plt.show()
plt.savefig("speedup_plot_v1.png", dpi=300)
print("✅ Gráfico guardado como speedup_plot.png")