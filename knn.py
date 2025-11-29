from mpi4py import MPI
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

# ============================================================
# MPI setup
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# KNN function
# ============================================================
def knn_predict(test_point, X_train, y_train, k):
    # Distancias euclidianas (sin sqrt, no afecta el top-k)
    diff = X_train - test_point
    distances = np.sum(diff **2, axis=1)
    # indices de los k vecinos mas cercanos
    k_idx = np.argpartition(distances, k)[:k]
    # voto mayoritario
    return Counter(y_train[k_idx]).most_common(1)[0][0]


# ============================================================
# Marca de tiempo total
# ============================================================
t_total_start = time.time()

# ============================================================
# Rank 0 carga MNIST desde mnist.npz y hace el split
# ============================================================
if rank == 0:
    print("Cargando MNIST local desde archivo mnist.npz...")

    data = np.load("mnist.npz")
    X = data["x_train"].reshape((-1, 784)).astype("float32") / 255.0
    y = data["y_train"].astype("int")

    print("Haciendo split de dataset (train / test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("SHAPES:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_train:", y_train.shape)
    print("  y_test :", y_test.shape)
else:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

# ============================================================
# Broadcast de los datos de entrenamiento
# (todos los procesos necesitan X_train e y_train)
# ============================================================
t_bcast_start = time.time()
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
t_bcast_end = time.time()

# ============================================================
# Scatter de los datos de test (X_test, y_test) entre procesos
# Cada proceso recibe un chunk diferente
# ============================================================
t_scatter_start = time.time()

if rank == 0:
    parts_X = np.array_split(X_test, size)
    parts_y = np.array_split(y_test, size)
else:
    parts_X = None
    parts_y = None

local_X_test = comm.scatter(parts_X, root=0)
local_y_test = comm.scatter(parts_y, root=0)

t_scatter_end = time.time()

# ============================================================
# Cada proceso calcula sus predicciones locales
# ============================================================
t_compute_start = time.time()

k = 3
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
local_predictions = np.array(local_predictions, dtype=int)

t_compute_end = time.time()

# ============================================================
# Gather de resultados hacia el rank 0
# ============================================================
t_gather_start = time.time()

all_predictions = comm.gather(local_predictions, root=0)
all_y_true = comm.gather(local_y_test, root=0)

t_gather_end = time.time()

# ============================================================
# Rank 0 arma todo y calcula metricas
# ============================================================
if rank == 0:
    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_y_true)

    accuracy = np.mean(y_pred == y_true)
    t_total_end = time.time()

    t_total = t_total_end - t_total_start
    t_comm = ((t_bcast_end - t_bcast_start)
              + (t_scatter_end - t_scatter_start)
              + (t_gather_end - t_gather_start))
    t_comp = (t_compute_end - t_compute_start)

    print("\n===== RESULTADOS DEL KNN PARALELO - MNIST =====")
    print(f"Procesos utilizados (p): {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tiempo Total: {t_total:.4f} s")
    print(f"Tiempo Comunicacion: {t_comm:.4f} s")
    print(f"Tiempo Computacion (KNN puro): {t_comp:.4f} s")

    # Guardar resultados en CSV
    with open("knn_times_v3.csv", "a") as f:
        f.write(f"{size},{accuracy:.4f},{t_total:.4f},{t_comp:.4f},{t_comm:.4f}\n")

    print("Resultados guardados en knn_times.csv")
