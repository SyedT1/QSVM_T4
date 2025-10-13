import pennylane as qml
from pennylane import numpy as pnp

class QuantumCircuits:
    """Defines quantum circuits for variational quantum classifiers."""
    
    def __init__(self, device_name="default.qubit", wires=2):
        self.dev = qml.device(device_name, wires=wires)
    
    def encode_features(self, x):
        """Angle encoding: map features to RY rotations."""
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
    
    def proposal1_ansatz(self, params):
        """2-layer ansatz with RY, RZ, and CNOT(0→1)."""
        # Layer 1
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        qml.RY(params[2], wires=1)
        qml.RZ(params[3], wires=1)
        qml.CNOT(wires=[0, 1])
        
        # Layer 2
        qml.RY(params[4], wires=0)
        qml.RZ(params[5], wires=0)
        qml.RY(params[6], wires=1)
        qml.RZ(params[7], wires=1)
        qml.CNOT(wires=[0, 1])
    
    def proposal2_ansatz(self, params):
        """3-layer ansatz with RX, RY, and CNOT(1→0)."""
        param_idx = 0
        for layer in range(3):
            qml.RX(params[param_idx], wires=0)
            qml.RY(params[param_idx + 1], wires=0)
            qml.RX(params[param_idx + 2], wires=1)
            qml.RY(params[param_idx + 3], wires=1)
            param_idx += 4
            qml.CNOT(wires=[1, 0])
    
    def proposal1_circuit(self, params, x):
        """Circuit for Proposal 1: encoding + ansatz + measurement on qubit 1."""
        self.encode_features(x)
        self.proposal1_ansatz(params)
        return qml.expval(qml.PauliZ(1))
    
    def proposal2_circuit(self, params, x):
        """Circuit for Proposal 2: encoding + ansatz + measurement on qubit 0."""
        self.encode_features(x)
        self.proposal2_ansatz(params)
        return qml.expval(qml.PauliZ(0))
    
    def get_qnodes(self):
        """Return QNodes for both proposals."""
        return (
            qml.QNode(self.proposal1_circuit, self.dev),
            qml.QNode(self.proposal2_circuit, self.dev)
        )