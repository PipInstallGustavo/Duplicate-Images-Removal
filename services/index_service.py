import os
import numpy as np
from .clip_service import CLIPService
from .pca_service import PCAService
from .milvus_service import MilvusService

class IndexService:
    def __init__(self, folder: str, collection_name="image_clip", dim_pca=128, batch_size=200):
        self.folder = folder
        self.collection_name = collection_name
        self.dim_pca = dim_pca
        self.batch_size = batch_size

        self.clip_service = CLIPService()
        self.pca_service = PCAService()
        self.milvus_service = MilvusService()
        self.pca_model_path = self.pca_service.model_path
        self.collection = None

    def index_images(self):
        embeddings, paths = [], []
        print("Iniciando indexação de imagens...")

        for file in os.listdir(self.folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
                path = os.path.join(self.folder, file)
                emb = self.clip_service.get_embedding(path)
                embeddings.append(emb)
                paths.append(path)
                print(f"Imagem processada: {file}")

        embeddings = np.array(embeddings)

        #  Treina ou transforma PCA dependendo da existência do modelo 
        if os.path.exists(self.pca_model_path):
            print("Usando PCA existente para transformar embeddings...")
            reduced = self.pca_service.transform(embeddings)
        else:
            print("Nenhum PCA encontrado - treinando novo modelo.")
            reduced = self.pca_service.train(embeddings)

        collection= self.milvus_service.create_collection(self.collection_name, self.dim_pca)

        # Inserção em lotes
        for i in range(0, len(paths), self.batch_size):
            batch_emb = reduced[i:i+self.batch_size].tolist()
            batch_paths = paths[i:i+self.batch_size]
            collection.insert([batch_emb, batch_paths])

        collection.flush()
        collection.load()
        print(f"{len(paths)} imagens indexadas em '{self.collection_name}'.")
        self.collection = collection
        return self.collection, self.pca_service

