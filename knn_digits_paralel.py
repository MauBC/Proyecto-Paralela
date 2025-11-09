from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def knn_predict(test_point, X_train, y_train, k):
    # Distancias vectorizadas (más rápido que la versión secuencial)
    distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    return Counter(k_labels).most_common(1)[0][0]

# Solo el proceso 0 carga los datos
if rank == 0:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    start = time.time()
else:
    X_train = X_test = y_train = y_test = None

# Enviar el conjunto de entrenamiento a todos
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)

# Dividir X_test en partes (scatter)
X_test_parts = np.array_split(X_test, size) if rank == 0 else None
local_X_test = comm.scatter(X_test_parts, root=0)

# Cada proceso clasifica su parte
k = 3
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_X_test]

# Juntar predicciones
all_predictions = comm.gather(local_predictions, root=0)

# Proceso 0 calcula accuracy
if rank == 0:
    y_pred = np.concatenate(all_predictions)
    accuracy = np.mean(y_pred == y_test)
    end = time.time()
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"⏱️ Tiempo paralelizado: {end - start:.4f} segundos")
