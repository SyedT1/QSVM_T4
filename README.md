# QSVM_T4

## 1. Alternating Entanglement Feature Map (AEFM)

### Circuit Structure (2 layers):

- **Initial encoding:**
  $$R_{y}(x_1) \otimes R_{y}(x_2)$$

- **Layer 1:**
  - $$R_{y}(x_1) \otimes R_{y}(x_2)$$
  - $$\mathrm{CNOT}_{0\to1}$$
  - $$R_{z}(x_1) \otimes R_{z}(x_2)$$
  - $$\mathrm{CNOT}_{1\to0}$$

- **Layer 2:**
  - $$R_{y}(x_1) \otimes R_{y}(x_2)$$
  - $$\mathrm{CNOT}_{0\to1}$$
  - $$R_{z}(x_1) \otimes R_{z}(x_2)$$
  - $$\mathrm{CNOT}_{1\to0}$$

### Quantum Feature State:
$$|\phi_\text{AEFM}(x)\rangle = U_\text{AEFM}(x)|00\rangle$$

### Kernel Function:
$$k_\text{AEFM}(\mathbf{x}, \mathbf{x'}) = |\langle\phi_\text{AEFM}(\mathbf{x})|\phi_\text{AEFM}(\mathbf{x'})\rangle|^2$$

## 2. Strongly Entangling Feature Map with Hadamard (SEFM-H)

### Circuit Structure (3 identical layers):

- **Initial encoding:**
  $$R_{y}(x_1) \otimes R_{y}(x_2)$$

- **Each layer ℓ = 1,2,3:**
  - $$H \otimes H$$
  - $$R_{y}(x_1) \otimes R_{y}(x_2)$$
  - $$\mathrm{CNOT}_{0\to1}$$
  - $$R_{z}(x_1) \otimes R_{z}(x_2)$$

### Quantum Feature State:
$$|\phi_\text{SEFM-H}(x)\rangle = U_\text{SEFM-H}(x)|00\rangle$$

### Kernel Function:
$$k_\text{SEFM-H}(\mathbf{x}, \mathbf{x'}) = |\langle\phi_\text{SEFM-H}(\mathbf{x})|\phi_\text{SEFM-H}(\mathbf{x'})\rangle|^2$$



