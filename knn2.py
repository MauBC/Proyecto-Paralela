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
# Optimized batch KNN function
# ============================================================
def knn_batch_predict(X_test_batch, X_train, y_train, k):
    # Compute squared L2 distances using vectorized formula:
    # ||A - B||^2 = |A|^2 + |B|^2 - 2*A.B
    A2 = np.sum(X_test_batch ** 2, axis=1).reshape(-1, 1)
    B2 = np.sum(X_train ** 2, axis=1).reshape(1, -1)
    AB = X_test_batch @ X_train.T

    dists = A2 + B2 - 2 * AB

    # k nearest neighbors
    idx = np.argpartition(dists, k, axis=1)[:, :k]

    # majority vote
    preds = []
    for row in idx:
        preds.append(Counter(y_train[row]).most_common(1)[0][0])

    return np.array(preds, dtype=int)

# ============================================================
# Time marker
# ============================================================
t_total_start = time.time()

# ============================================================
# Rank 0 loads MNIST (local npz file)
# ============================================================
if rank == 0:
    print("Loading MNIST from mnist.npz...")

    data = np.load("mnist.npz")
    X = data["x_train"].reshape((-1, 784)).astype("float32") / 255.0
    y = data["y_train"].astype(int)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("SHAPES:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

else:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

# ============================================================
# Broadcast training data
# ============================================================
t_bcast_start = time.time()
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)
t_bcast_end = time.time()

# ============================================================
# Scatter test data among processes
# ============================================================
t_scatter_start = time.time()

parts_X = np.array_split(X_test, size) if rank == 0 else None
parts_y = np.array_split(y_test, size) if rank == 0 else None

local_X_test = comm.scatter(parts_X, root=0)
local_y_test = comm.scatter(parts_y, root=0)

t_scatter_end = time.time()

# ============================================================
# Compute local predictions using batches
# ============================================================
t_compute_start = time.time()

k = 3
batch_size = 200  # adjust to tune RAM usage and speed

local_predictions = []

for i in range(0, len(local_X_test), batch_size):
    batch = local_X_test[i:i+batch_size]
    preds = knn_batch_predict(batch, X_train, y_train, k)
    local_predictions.append(preds)

local_predictions = np.concatenate(local_predictions)

t_compute_end = time.time()

# ============================================================
# Gather predictions and true labels
# ============================================================
t_gather_start = time.time()

all_predictions = comm.gather(local_predictions, root=0)
all_y_true = comm.gather(local_y_test, root=0)

t_gather_end = time.time()

# ============================================================
# Final metrics (root)
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

    print("\n===== PARALLEL KNN RESULTS (OPTIMIZED) =====")
    print(f"Processes (p): {size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total Time: {t_total:.4f} s")
    print(f"Compute Time: {t_comp:.4f} s")
    print(f"Communication Time: {t_comm:.4f} s")

    # Save to CSV
    with open("knn_times_v4.csv", "a") as f:
        f.write(f"{size},{accuracy:.4f},{t_total:.4f},{t_comp:.4f},{t_comm:.4f}\n")

    print("Saved results to knn_times_v4.csv")
