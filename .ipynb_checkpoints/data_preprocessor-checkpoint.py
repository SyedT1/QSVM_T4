# data_preprocessor.py (updated)
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler  # Changed from StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreprocessor:
    def __init__(self, n_components=2, test_size=0.3, random_state=42):
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))  # Critical change!
    
    def load_and_preprocess(self):
        iris = load_iris()
        # Use Versicolor (1) and Virginica (2) - NOT Setosa!
        mask = (iris.target == 1) | (iris.target == 2)
        X = iris.data[mask]
        y = (iris.target[mask] == 2).astype(int)  # Virginica=1, Versicolor=0
        
        X_pca = self.pca.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_pca)  # Now scaled to [0, π]
        
        return train_test_split(
            X_scaled, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
    
    def get_explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_