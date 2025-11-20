from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from services.clip_service import CLIPService
from services.pca_service import PCAService
from services.milvus_service import MilvusService
from services.index_service import IndexService
from services.search_service import SearchService
from services.delete_service import DeleteService
import tempfile
import shutil
import os
import imghdr

router = APIRouter()

# --- Inicialização dos serviços ---
# Conecta ao Milvus e cria collection se não existir
milvus_service = MilvusService()

# milvus_service.drop_collection("image_clip") -- usado para debug

collection = milvus_service.create_collection("image_clip")


# Indexa imagens se a collection estiver vazia
# path da pasta das imagens
if collection.num_entities == 0:
    index_service = IndexService(os.path.join("mnt", "faroldigital-cloud", "media")) # mudar pasta na produção
    print("Collection vazia. Indexando imagens...")
    index_service.index_images()

# Inicializa PCA
pca_service = PCAService()
pca_service.load() # treina PCA se não existir

# Inicializa CLIP
clip_service = CLIPService()

# Inicializa serviços de busca e deleção
search_service = SearchService(collection, pca_service)
delete_service = DeleteService(collection, pca_service)


# --- Função utilitária para salvar arquivo temporário ---
def salvar_temporariamente(file: UploadFile) -> str:
    content = file.file.read()
    file.file.seek(0)

    detected_type = imghdr.what(None, h=content)
    extension_map = {
        "jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "jfif": ".jfif",
        "bmp": ".bmp", "tiff": ".tiff", "webp": ".webp",
    }
    file_extension = extension_map.get(detected_type, ".jfif")
    tmp_path = tempfile.mktemp(suffix=file_extension)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return tmp_path


# --- Rotas ---
@router.post("/search")
async def search(file: UploadFile = File(...), top_k: int = Form(10), threshold: float = Form(0.9)):
    try:
        tmp_path = salvar_temporariamente(file)
        results = search_service.search(tmp_path, top_k=top_k, threshold=threshold)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/delete-similar")
async def delete_similar(file: UploadFile = File(...), threshold: float = Form(...), top_k: int = Form(10)):
    try:
        result = delete_service.delete_similar(file.filename, threshold=threshold, top_k=top_k)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@router.get("/count_images") #contar todas as imagens do bd
def count_images():
    count = milvus_service.count_images(collection.name)
    return {"count": count}

