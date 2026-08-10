# ============================================================
# PART 1
# Batch Gradient Descent vs Stochastic GD vs Mini-Batch GD
# ============================================================

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. CREATE A LARGE DATASET
# ------------------------------------------------------------

X, y = make_regression(
    n_samples=100000,
    n_features=20,
    noise=10,
    random_state=42
)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column
X = np.c_[np.ones(X.shape[0]), X]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])
print("Features        :", X_train.shape[1])


# ------------------------------------------------------------
# 2. LOSS FUNCTION
# ------------------------------------------------------------

def mse_loss(X, y, w):
    predictions = X @ w
    return np.mean((predictions - y) ** 2)


# ------------------------------------------------------------
# 3. BATCH GRADIENT DESCENT
# ------------------------------------------------------------

def batch_gradient_descent(X, y, learning_rate=0.01, epochs=50):

    w = np.zeros(X.shape[1])

    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):

        # Predictions
        predictions = X @ w

        # Gradient using the complete dataset
        gradient = (2 / len(X)) * X.T @ (predictions - y)

        # Parameter update
        w = w - learning_rate * gradient

        # Calculate loss
        loss = mse_loss(X, y, w)
        loss_history.append(loss)

    execution_time = time.time() - start_time

    return w, loss_history, execution_time


# ------------------------------------------------------------
# 4. STOCHASTIC GRADIENT DESCENT
# ------------------------------------------------------------

def stochastic_gradient_descent(
    X,
    y,
    learning_rate=0.001,
    epochs=50
):

    w = np.zeros(X.shape[1])

    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):

        # Shuffle dataset
        indices = np.random.permutation(len(X))

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(len(X_shuffled)):

            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # Prediction for one sample
            prediction = xi @ w

            # Gradient for one sample
            gradient = 2 * xi * (prediction - yi)

            # Parameter update
            w = w - learning_rate * gradient

        # Calculate full training loss
        loss = mse_loss(X, y, w)
        loss_history.append(loss)

    execution_time = time.time() - start_time

    return w, loss_history, execution_time


# ------------------------------------------------------------
# 5. MINI-BATCH GRADIENT DESCENT
# ------------------------------------------------------------

def mini_batch_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=50,
    batch_size=64
):

    w = np.zeros(X.shape[1])

    loss_history = []

    start_time = time.time()

    for epoch in range(epochs):

        # Shuffle data
        indices = np.random.permutation(len(X))

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process batches
        for start in range(0, len(X), batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Predictions
            predictions = X_batch @ w

            # Batch gradient
            gradient = (
                (2 / len(X_batch))
                * X_batch.T
                @ (predictions - y_batch)
            )

            # Parameter update
            w = w - learning_rate * gradient

        # Full training loss
        loss = mse_loss(X, y, w)
        loss_history.append(loss)

    execution_time = time.time() - start_time

    return w, loss_history, execution_time


# ------------------------------------------------------------
# 6. TRAIN ALL THREE METHODS
# ------------------------------------------------------------

batch_w, batch_loss, batch_time = batch_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=50
)

sgd_w, sgd_loss, sgd_time = stochastic_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.001,
    epochs=50
)

mini_w, mini_loss, mini_time = mini_batch_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=50,
    batch_size=64
)


# ------------------------------------------------------------
# 7. RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GRADIENT DESCENT COMPARISON")
print("=" * 60)

print(
    f"Batch GD       | Time: {batch_time:.4f} sec "
    f"| Final Loss: {batch_loss[-1]:.6f}"
)

print(
    f"SGD            | Time: {sgd_time:.4f} sec "
    f"| Final Loss: {sgd_loss[-1]:.6f}"
)

print(
    f"Mini-Batch GD  | Time: {mini_time:.4f} sec "
    f"| Final Loss: {mini_loss[-1]:.6f}"
)


# ------------------------------------------------------------
# 8. PLOT LOSS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(batch_loss, label="Batch GD")
plt.plot(sgd_loss, label="SGD")
plt.plot(mini_loss, label="Mini-Batch GD (64)")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Batch GD vs SGD vs Mini-Batch GD")
plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 9. IDENTIFY BEST METHOD
# ------------------------------------------------------------

methods = {
    "Batch GD": (batch_time, batch_loss[-1]),
    "SGD": (sgd_time, sgd_loss[-1]),
    "Mini-Batch GD": (mini_time, mini_loss[-1])
}

print("\nObservation:")
print("------------")

fastest = min(methods, key=lambda x: methods[x][0])

print("Fastest execution:", fastest)

print("""
For large-scale datasets, Mini-Batch Gradient Descent is generally
the most suitable choice.

Reason:
1. Batch GD calculates the gradient using the entire dataset.
2. SGD updates parameters after every single sample, causing noisy
   updates and many parameter-update operations.
3. Mini-Batch GD processes a small batch at a time.
4. Batch size 64 provides a good balance between computational
   efficiency and stable gradient estimation.
5. Mini-Batch GD can efficiently use vectorized CPU/GPU operations.

Therefore, Mini-Batch Gradient Descent is usually preferred for
large-scale machine learning datasets.
""")