from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreprocessor:
    """Handles loading, filtering, and preprocessing of the Iris dataset for binary classification."""
    
    def __init__(self, n_components=2, test_size=0.3, random_state=42):
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
    
    def load_and_preprocess(self):
        """Load Iris dataset, filter for binary classification, apply PCA and standardization."""
        # Load and filter data for Setosa (0) and Versicolor (1)
        iris = load_iris()
        mask = (iris.target != 2)  # Binary classification
        X = iris.data[mask]  # Shape: (100, 4)
        y = iris.target[mask]  # Shape: (100,)
        
        # Apply PCA to reduce to n_components
        X_pca = self.pca.fit_transform(X)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_pca)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_explained_variance_ratio(self):
        """Return the explained variance ratio of PCA."""
        return self.pca.explained_variance_ratio_