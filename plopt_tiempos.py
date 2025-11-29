import pandas as pd
import matplotlib.pyplot as plt

# Cargar resultados desde tu CSV
df = pd.read_csv("knn_times_v1.2.csv", header=None,
                 names=["p","accuracy","total","compute","comm"])

# Gráfico de líneas: computación vs comunicación
plt.figure(figsize=(8,6))
plt.plot(df["p"], df["compute"], marker="o", label="Computación (KNN)", color="steelblue")
plt.plot(df["p"], df["comm"], marker="s", label="Comunicación", color="orange")
plt.plot(df["p"], df["total"], marker="^", label="Tiempo Total", color="green")

plt.xlabel("Número de procesos (p)")
plt.ylabel("Tiempo [s]")
plt.title("Evolución de tiempos según número de procesos")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Guardar la gráfica como imagen
plt.savefig("knn_tiempos_v1.2.png", dpi=300, bbox_inches="tight")
