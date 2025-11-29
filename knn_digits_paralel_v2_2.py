from mpi4py import MPI
from collections import Counter
import numpy as np
import time

def knn_predict(test_point, X_train, y_train, k):
    distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
    k_idx = np.argpartition(distances, k)[:k]
    return Counter(y_train[k_idx]).most_common(1)[0][0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
t_total_start = time.time()

# Solo el proceso 0 carga los datos
if rank == 0:
    # Leer MNIST desde archivo local
    data = np.load("mnist2.npz")
    X_train, y_train = data["x_train"], data["y_train"]
    X_test, y_test = data["x_test"], data["y_test"]

    # Normalizar y reestructurar
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
else:
    X_train = X_test = y_train = y_test = None

# Enviar el conjunto de entrenamiento a todos
t_bcast_start = time.time()
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
t_bcast_end = time.time()

# Dividir X_test en partes (scatter)
t_scatter_start = time.time()
if rank == 0:
    X_test_parts = np.array_split(X_test, size)
else:
    X_test_parts = None
local_X_test = comm.scatter(X_test_parts, root=0)
t_scatter_end = time.time()

# Computar predicciones locales
t_compute_start = time.time()
k = 3
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
t_compute_end = time.time()

# Juntar predicciones
t_gather_start = time.time()
all_predictions = comm.gather(local_predictions, root=0)
t_gather_end = time.time()

# Evaluación final (solo en master)
if rank == 0:
    y_pred = np.concatenate(all_predictions)
    accuracy = np.mean(y_pred == y_test)
    t_total_end = time.time()

    print("\n===== RESULTADOS DEL KNN PARALELO - V2.2 =====")
    print(f"Procesos utilizados (p): {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tiempo Total: {t_total_end - t_total_start:.4f} s")
    print(f"Tiempo Comunicación: {(t_bcast_end - t_bcast_start) + (t_scatter_end - t_scatter_start) + (t_gather_end - t_gather_start):.4f} s")
    print(f"Tiempo Computación (KNN puro): {t_compute_end - t_compute_start:.4f} s")

    with open("knn_times_v2.2.med.csv", "a") as f:
        f.write(f"{size},{accuracy:.4f},{t_total_end - t_total_start:.4f},{t_compute_end - t_compute_start:.4f},{(t_bcast_end - t_bcast_start) + (t_scatter_end - t_scatter_start) + (t_gather_end - t_gather_start):.4f}\n")

    print("Resultados guardados en knn_times_v2_2.csv\n")
