import numpy as np

# Cargar el archivo original
data = np.load("mnist2.npz")

X_train = data["x_train"]
y_train = data["y_train"]
X_test  = data["x_test"]
y_test  = data["y_test"]

# Porcentaje que quieres conservar (20%)
ratio = 0.5

# Calcular las nuevas longitudes
new_train_size = int(len(X_train) * ratio)
new_test_size  = int(len(X_test) * ratio)

print("Nuevo tamaño train:", new_train_size)
print("Nuevo tamaño test:", new_test_size)

# Recortar el dataset
X_train_small = X_train[:new_train_size]
y_train_small = y_train[:new_train_size]
X_test_small  = X_test[:new_test_size]
y_test_small  = y_test[:new_test_size]

# Guardar nuevo archivo reducido
np.savez(
    "mnist.npz",
    x_train=X_train_small,
    y_train=y_train_small,
    x_test=X_test_small,
    y_test=y_test_small
)

print("Archivo reducido guardado como mnist_reducido_20.npz")
