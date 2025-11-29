from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import time

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Cargar y dividir los datos
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

k = 3

t_total_start = time.time()

# Realizar predicciones
t_compute_start = time.time()
y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]
t_compute_end = time.time()

# Evaluar
accuracy = np.mean(y_pred == y_test)
t_total_end = time.time()

# --- Resultados ---
print("\n===== RESULTADOS DEL KNN SECUENCIAL =====")
print(f"Procesos utilizados (p): 1")
print(f"Accuracy: {accuracy:.4f}")
print(f"Tiempo Total: {t_total_end - t_total_start:.4f} s")
print(f"Tiempo Computación (KNN puro): {t_compute_end - t_compute_start:.4f} s")
print(f"Tiempo Comunicación: 0.0000 s (no aplica en secuencial)")

with open("knn_time_seq.csv", "a") as f:
    f.write(f"1,{accuracy:.4f},{t_total_end - t_total_start:.4f},{t_compute_end - t_compute_start:.4f},0.0000\n")

print("Resultados guardados en knn_time_seq.csv\n")