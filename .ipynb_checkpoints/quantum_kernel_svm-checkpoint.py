import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class QuantumSVM:
    """Trains classical SVM with quantum kernel."""
    
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
    
    def compute_kernel_matrix(self, X1, X2=None):
        """Compute quantum kernel matrix K_ij = |<φ(x_i)|φ(x_j)>|^2."""
        if X2 is None:
            X2 = X1
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i, j] = self.kernel_fn(X1[i], X2[j])
        return K
    
    def evaluate(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """Train and evaluate SVM with quantum kernel."""
        K_train = self.compute_kernel_matrix(X_train)
        K_test = self.compute_kernel_matrix(X_test, X_train)
        
        svm = SVC(kernel='precomputed')
        svm.fit(K_train, y_train)
        
        train_acc = svm.score(K_train, y_train)
        test_acc = svm.score(K_test, y_test)
        cv_scores = cross_val_score(svm, K_train, y_train, cv=cv_folds)
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }