import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("knn_times.csv", delimiter=",")
p = data[:,0]
t_total = data[:,2]

t1 = t_total[0]
speedup = t1 / t_total
efficiency = speedup / p

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(p, speedup, marker="o")
plt.title("Speedup vs Procesos")
plt.xlabel("Procesos (p)")
plt.ylabel("Speedup (S)")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(p, efficiency, marker="o", color="orange")
plt.title("Eficiencia vs Procesos")
plt.xlabel("Procesos (p)")
plt.ylabel("Eficiencia (S/p)")
plt.grid(True)

plt.tight_layout()
plt.show()
