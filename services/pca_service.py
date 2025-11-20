import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# from .index_service import IndexService

class PCAService:
    def __init__(self, model_path="pca_model.joblib", dim_pca=128):
        self.model_path = model_path
        self.dim_pca = dim_pca
        self.pca_model = None

        if os.path.exists(self.model_path):
            self.load()
        else:
            print("Nenhum modelo PCA encontrado. Você precisa treiná-lo com 'train()'.")
        

    def train(self, embeddings: np.ndarray):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        self.pca_model = PCA(n_components=self.dim_pca, svd_solver="auto", random_state=42)
        reduced = self.pca_model.fit_transform(embeddings)
        joblib.dump(self.pca_model, self.model_path)
        print(f"PCA treinado e salvo em: {self.model_path}")
        return reduced.astype(np.float32)

    def load(self):
        if os.path.exists(self.model_path):
            self.pca_model = joblib.load(self.model_path)
            print(f"PCA carregado de: {self.model_path}")
        else:
            print("PCA não encontrado, treine com 'train()'.")
        return self.pca_model

    def transform(self, embeddings: np.ndarray):
        if self.pca_model is None:
            raise ValueError("Modelo PCA não carregado ou treinado.")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        reduced = self.pca_model.transform(embeddings)
        reduced = normalize(reduced, norm="l2")
        return reduced.astype(np.float32)
