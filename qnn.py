"""
Qiskit QNN for classifying MNIST digits (binary subset: 0s and 1s).

Why binary + downsampled?
--------------------------
Full MNIST is 28x28 = 784 pixels and 10 classes. Current-generation QNNs
(simulated or on real hardware) can only realistically handle a handful of
qubits (data-encoding qubits are usually in the single digits, sometimes
up to ~16-20 on a simulator, far fewer on real hardware). Feeding 784
features directly into a quantum circuit is not feasible with today's
methods, so this script:

  1. Uses only two digit classes (0 and 1) -> binary classification.
  2. Downsamples each image (e.g., to 4x4 = 16 pixels via PCA or resizing)
     so it fits on a small number of qubits.
  3. Encodes those features with a feature map, then trains a
     variational (parameterized) ansatz as the trainable "neural network".

If you need full 10-class MNIST at native resolution, a QNN is not a good
fit today -- use a classical CNN instead, or use a hybrid approach where
a classical CNN does most of the heavy lifting and only a small
bottleneck layer is quantum (shown in the "hybrid" note at the bottom).

Install:
    pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn torch torchvision
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------
# 1. Load MNIST and restrict to two classes (0 and 1)
# ---------------------------------------------------------------------
print("Loading MNIST (this can take a minute the first time)...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

# Use a small subset for speed (quantum simulation is slow)
X, _, y, _ = train_test_split(X, y, train_size=400, stratify=y, random_state=42)

# ---------------------------------------------------------------------
# 2. Reduce dimensionality to fit on N qubits (here: 4 qubits -> 4 features)
# ---------------------------------------------------------------------
N_QUBITS = 4

pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X)

scaler = MinMaxScaler((0, np.pi))  # scale into a range good for angle encoding
X_scaled = scaler.fit_transform(X_reduced)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ---------------------------------------------------------------------
# 3. Build the quantum circuit: feature map (data encoding) + ansatz (trainable)
# ---------------------------------------------------------------------
feature_map = ZFeatureMap(feature_dimension=N_QUBITS, reps=1)
ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=2)

qc = feature_map.compose(ansatz)

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)

# Wrap the QNN so it behaves like a PyTorch layer
quantum_layer = TorchConnector(qnn)

# ---------------------------------------------------------------------
# 4. Full hybrid model: quantum layer -> linear -> sigmoid
# ---------------------------------------------------------------------
class HybridQNN(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()
        self.qlayer = quantum_layer
        self.linear = nn.Linear(1, 1)  # EstimatorQNN default output dim = 1

    def forward(self, x):
        x = self.qlayer(x)
        x = self.linear(x)
        return torch.sigmoid(x)


model = HybridQNN(quantum_layer)

# ---------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

EPOCHS = 20
print("Training...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = loss_fn(output, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            preds = (model(X_test_t) > 0.5).float()
            acc = (preds == y_test_t).float().mean().item()
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | loss={loss.item():.4f} | test_acc={acc:.3f}")

# ---------------------------------------------------------------------
# 6. Final evaluation
# ---------------------------------------------------------------------
with torch.no_grad():
    preds = (model(X_test_t) > 0.5).float()
    acc = (preds == y_test_t).float().mean().item()
print(f"\nFinal test accuracy: {acc:.3f}")

