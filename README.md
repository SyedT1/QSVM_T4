# QSVM_T4

## 1. Alternating Entanglement Feature Map (AEFM)

### Circuit Structure (2 layers):

- **Initial encoding:**
  $$R_{y}(x_1) \otimes R_{y}(x_2)$$

- **Layer 1:**
  $$
  \begin{aligned}
  & R_{y}(x_1) \otimes R_{y}(x_2) \\
  & \mathrm{CNOT}_{0\to1} \\
  & R_{z}(x_1) \otimes R_{z}(x_2) \\
  & \mathrm{CNOT}_{1\to0}
  \end{aligned}
  $$

- **Layer 2:**
  $$
  \begin{aligned}
  & R_{y}(x_1) \otimes R_{y}(x_2) \\
  & \mathrm{CNOT}_{0\to1} \\
  & R_{z}(x_1) \otimes R_{z}(x_2) \\
  & \mathrm{CNOT}_{1\to0}
  \end{aligned}
  $$

### Quantum Feature State:
$$|\phi_\text{AEFM}(x)\rangle = U_\text{AEFM}(x)|00\rangle$$

### Kernel Function:
$$k_\text{AEFM}(\mathbf{x}, \mathbf{x'}) = |\langle\phi_\text{AEFM}(\mathbf{x})|\phi_\text{AEFM}(\mathbf{x'})\rangle|^2$$


