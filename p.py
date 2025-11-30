import numpy as np

# Carga datos: size, accuracy, t_total, t_compute, t_comm
data = np.loadtxt("v2_khipu/knn_times_v2.2.med_khipu.csv", delimiter=",")
p = data[:,0].astype(int)
t_total = data[:,2]
t_compute = data[:,3]
t_comm = data[:,4]

# Ordenar por p
idx = np.argsort(p)
p, t_total, t_compute, t_comm = p[idx], t_total[idx], t_compute[idx], t_comm[idx]

# Speedup y eficiencia
if 1 not in p: raise ValueError("Falta medición con p=1")
T1 = t_compute[p==1][0]
speedup = T1 / t_compute
efficiency = speedup / p
comm_ratio = t_comm / t_total

# FLOPs/s (ajusta N_train, M, N_test)
N_train, M, N_test = 30000, 784, 5000  # ejemplo MNIST
FLOPS_TOTAL = N_test * N_train * (3 * M)
flops_per_sec = FLOPS_TOTAL / t_compute

# Selecciones
p_throughput = p[np.argmax(flops_per_sec)]
p_min_time = p[np.argmin(t_total)]

# Umbrales
mask = (efficiency >= 0.65) & (comm_ratio <= 0.30)
p_thresholded = p[mask]
p_eff_comm = p_thresholded[0] if p_thresholded.size else p_min_time

print(f"p óptimo por FLOPs/s: {p_throughput}")
print(f"p óptimo por tiempo total: {p_min_time}")
print(f"p recomendado (umbrales eficiencia/comunicación): {p_eff_comm}")