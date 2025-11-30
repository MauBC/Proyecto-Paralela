import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parámetros del dataset
N_train = 1437   # MNIST: 60000, Digits: 1437
M = 64           # MNIST: 784, Digits: 64
N_test = 360    # MNIST: 10000, Digits: 360

FLOPS_TOTAL = N_test * N_train * (3 * M)

data = np.loadtxt("v1_khipu/knn_times_v1_khipu.csv", delimiter=",")
p = data[:,0].astype(int)
t_compute = data[:,3]

idx = np.argsort(p)
p = p[idx]
t_compute = t_compute[idx]

flops_per_sec = FLOPS_TOTAL / t_compute

plt.figure(figsize=(6,5))
plt.plot(p, flops_per_sec, marker="s", color="darkgreen")
plt.title("FLOPs por segundo vs Procesos")
plt.xlabel("Procesos (p)")
plt.ylabel("FLOPs/s")
plt.grid(True)
plt.tight_layout()
plt.savefig("flops_vs_p_v1_khipu.png", dpi=300)
print("✅ Gráfico guardado en flops_vs_p.png")
