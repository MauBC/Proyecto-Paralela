from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from collections import Counter
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def knn_predict(test_point, X_train, y_train, k):
    distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
    k_idx = np.argsort(distances)[:k]
    return Counter(y_train[k_idx]).most_common(1)[0][0]

# --- Time markers ---
t_total_start = time.time()

# Master loads data
if rank == 0:
    print("📥 Descargando MNIST (una sola vez, puede tardar unos segundos)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')

    print("✂️ Haciendo split de dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train = X_test = y_train = y_test = None

# Broadcast training data
t_bcast_start = time.time()
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
t_bcast_end = time.time()

# Scatter test data
t_scatter_start = time.time()
parts = np.array_split(X_test, size) if rank == 0 else None
local_X_test = comm.scatter(parts, root=0)
t_scatter_end = time.time()

# Compute predictions
t_compute_start = time.time()
k = 3
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]
t_compute_end = time.time()

# Gather results
t_gather_start = time.time()
all_predictions = comm.gather(local_predictions, root=0)
t_gather_end = time.time()

# Final assembly and metrics
if rank == 0:
    y_pred = np.concatenate(all_predictions)
    accuracy = np.mean(y_pred == y_test)
    t_total_end = time.time()

    print("\n===== RESULTADOS DEL KNN PARALELO - MNIST =====")
    print(f"Procesos utilizados (p): {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tiempo Total: {t_total_end - t_total_start:.4f} s")
    print(f"Tiempo Comunicación: {(t_bcast_end - t_bcast_start) + (t_scatter_end - t_scatter_start) + (t_gather_end - t_gather_start):.4f} s")
    print(f"Tiempo Computación (KNN puro): {t_compute_end - t_compute_start:.4f} s")

    with open("knn_times_2.csv", "a") as f:
        f.write(f"{size},{accuracy:.4f},{t_total_end - t_total_start:.4f},{t_compute_end - t_compute_start:.4f},{(t_bcast_end - t_bcast_start) + (t_scatter_end - t_scatter_start) + (t_gather_end - t_gather_start):.4f}\n")

    print("✅ Resultados guardados en knn_times.csv\n")
