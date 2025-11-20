import os
import threading
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Rotas
from routes.routes import router

# Serviços
from services.watchdog_service import WatchdogService
from services.milvus_service import MilvusService
from services.index_service import IndexService
from services.pca_service import PCAService
from services.clip_service import CLIPService


# ============================================================
#  CONFIGURAÇÕES INICIAIS DO SISTEMA
# ============================================================

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
MEDIA_DIR = os.path.join(BASE_DIR, "mnt", "faroldigital-cloud", "media")# pasta com as imagens - mudar na produção
FRONT_DIR = os.path.join(BASE_DIR, "front")

# Garantir que diretórios existem
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(FRONT_DIR, exist_ok=True)


# Inicializar
print("\n========================================")
print("  Inicializando Sistema de Similaridade")
print("========================================\n")

# Conectar ao Milvus 
milvus_service = MilvusService()
collection = milvus_service.create_collection("image_clip", dim_pca=128)

# PCA 
pca_service = PCAService()
pca_service.load()

# CLIP
clip_service = CLIPService()

# Indexação inicial se a collection estiver vazia 
if collection.num_entities == 0:
    print("Collection vazia — iniciando indexação inicial...")
    indexer = IndexService(MEDIA_DIR, collection_name="image_clip")
    indexer.index_images()
else:
    print(f"Coleção do Milvus já possui {collection.num_entities} embeddings.")


# watchdog
def start_watchdog():
    print("\n──────────────────────────────────────────────")
    print("  Iniciando Watchdog para monitorar novas imagens...")
    print("──────────────────────────────────────────────\n")

    watchdog = WatchdogService(
        folder_path=MEDIA_DIR,
        collection_name="image_clip",
        dim_pca=128
    )
    watchdog.start()


# Criar thread para o watchdog 
thread_watchdog = threading.Thread(target=start_watchdog, daemon=True)
thread_watchdog.start()

# Rotas da API
app.include_router(router)

# Servir imagens
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Servir frontend
app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")



# RODAR


# Para rodar:
# uvicorn main:app --host 0.0.0.0 --port 8000

print("\n========================================")
print("🚀 Sistema iniciado com sucesso!")
print("🌐 API rodando em /")
print("📁 Monitorando novas imagens em:", MEDIA_DIR)
print("========================================\n")
