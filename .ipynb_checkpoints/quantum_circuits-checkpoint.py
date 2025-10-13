# quantum_circuits.py
import pennylane as qml
import numpy as np

class QuantumCircuits:
    """Defines quantum feature maps for QSVM kernel computation."""
    
    def __init__(self, wires=2):
        self.wires = wires
    
    def aefm_feature_map(self, x):
        """Alternating Entanglement Feature Map (AEFM) - Proposal 1."""
        # Initial encoding
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        
        # Layer 1
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(x[0], wires=0)
        qml.RZ(x[1], wires=1)
        qml.CNOT(wires=[1, 0])
        
        # Layer 2
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(x[0], wires=0)
        qml.RZ(x[1], wires=1)
        qml.CNOT(wires=[1, 0])
    
    def sefmh_feature_map(self, x):
        """Strongly Entangling Feature Map with Hadamard (SEFM-H) - Proposal 2."""
        # Initial encoding
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        
        for _ in range(3):  # 3 layers
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.RZ(x[0], wires=0)
            qml.RZ(x[1], wires=1)
    
    def get_state_overlap(self, x, y, feature_map_fn):
        """Compute |<φ(x)|φ(y)>|^2 using state vectors (simulator only)."""
        dev = qml.device("default.qubit", wires=self.wires)
        
        @qml.qnode(dev)
        def circuit_x():
            feature_map_fn(x)
            return qml.state()
        
        @qml.qnode(dev)
        def circuit_y():
            feature_map_fn(y)
            return qml.state()
        
        state_x = circuit_x()
        state_y = circuit_y()
        return np.abs(np.vdot(state_x, state_y))**2  # |<x|y>|^2
    
    def get_kernel_functions(self):
        """Return kernel functions for both proposals."""
        def kernel1(x, y):
            return self.get_state_overlap(x, y, self.aefm_feature_map)
        
        def kernel2(x, y):
            return self.get_state_overlap(x, y, self.sefmh_feature_map)
        
        return kernel1, kernel2