import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
from data_preprocessor import DataPreprocessor
from quantum_circuits import QuantumCircuits

class QuantumClassifier:
    """Trains and evaluates variational quantum classifiers."""
    
    def __init__(self, qnode, n_params, lr=0.01, epochs=100, patience=15):
        self.qnode = qnode
        self.n_params = n_params
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.opt = qml.AdamOptimizer(stepsize=lr)
    
    def quantum_model(self, params, x):
        """Convert quantum expectation value to probability."""
        expval = self.qnode(params, x)
        return (expval + 1) / 2
    
    def binary_cross_entropy(self, predictions, targets):
        """Binary cross-entropy loss with numerical stability."""
        epsilon = 1e-15
        predictions = pnp.clip(predictions, epsilon, 1 - epsilon)
        return -pnp.mean(targets * pnp.log(predictions) + (1 - targets) * pnp.log(1 - predictions))
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the quantum model and return metrics."""
        params = pnp.array(pnp.random.uniform(-pnp.pi, pnp.pi, self.n_params), requires_grad=True)
        train_losses = []
        train_accs = []
        test_accs = []
        best_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training step
            def cost(params):
                predictions = pnp.array([self.quantum_model(params, x) for x in X_train])
                return self.binary_cross_entropy(predictions, y_train)
            
            params, current_loss = self.opt.step_and_cost(cost, params)
            train_losses.append(float(current_loss))
            
            # Calculate accuracies
            train_preds = np.array([float(self.quantum_model(params, x)) for x in X_train]) > 0.5
            train_acc = np.mean(train_preds == y_train)
            train_accs.append(train_acc)
            
            test_preds = np.array([float(self.quantum_model(params, x)) for x in X_test]) > 0.5
            test_acc = np.mean(test_preds == y_test)
            test_accs.append(test_acc)
            
            # Early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
        
        return {
            'params': params,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_epoch': epoch + 1,
            'best_test_acc': best_test_acc
        }
