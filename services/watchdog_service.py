import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .clip_service import CLIPService
from .pca_service import PCAService
from .milvus_service import MilvusService


class WatchdogService(FileSystemEventHandler):
    def __init__(self, folder_path: str, collection_name="image_clip", dim_pca=128):
        self.folder_path = folder_path
        self.collection_name = collection_name
        self.dim_pca = dim_pca

        # Serviços
        self.clip_service = CLIPService()
        self.pca_service = PCAService()
        self.milvus_service = MilvusService()

        # Carrega/Cria collection
        self.collection = self.milvus_service.create_collection(
            collection_name=self.collection_name,
            dim_pca=self.dim_pca
        )

        # Carrega PCA existente
        self.pca_service.load()

        print(f"Watchdog monitorando a pasta: {self.folder_path}")


    # Quando um arquivo novo é criado
    def on_created(self, event):
        if event.is_directory:
            return

        path = event.src_path.lower()
        if not path.endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
            return

        print(f"\nNova imagem detectada: {event.src_path}")

        # pequena espera p/ SO terminar de escrever o arquivo
        time.sleep(0.5)

        self.index_new_image(event.src_path)


    # Função que indexa uma nova imagem individual
    def index_new_image(self, img_path: str):
        try:
            print(f"Extraindo embedding (CLIP)...")
            emb = self.clip_service.get_embedding(img_path)

            print(f"Reduzindo para PCA ({self.dim_pca} dims)...")
            emb_reduced = self.pca_service.transform([emb])[0].tolist()

            print(f"Inserindo no Milvus...")
            self.collection.insert([[emb_reduced], [img_path]])
            self.collection.flush()

            print(f"Imagem '{os.path.basename(img_path)}' inserida com sucesso!")

        except Exception as e:
            print(f"Erro ao indexar imagem '{img_path}': {e}")


    # Inicia o Watchdog
    def start(self):
        observer = Observer()
        observer.schedule(self, self.folder_path, recursive=False)
        observer.start()

        print("Watchdog ativado. Monitorando alterações...\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()
