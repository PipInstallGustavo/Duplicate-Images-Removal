from fastapi import APIRouter, UploadFile, File, Form, Path
from fastapi.responses import JSONResponse
from clip_pca import search_similar, deletar_por_similaridade
import tempfile
import shutil
import os
import imghdr

app = APIRouter()

# rota de busca
@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = Form(10), threshold: float = Form(0.90)):
    # Lê o conteúdo do arquivo para detectar o tipo real da imagem
    content = await file.read()

    # Volta o ponteiro do arquivo para o início (para salvar depois)
    await file.seek(0)

    # Detecta o tipo de imagem a partir do conteúdo binário
    detected_type = imghdr.what(None, h=content)

    if not detected_type:
        return {"status": "error", "message": "O arquivo enviado não é uma imagem válida."}

    # Mapeamento dos tipos detectados para extensões de arquivo
    extension_map = {
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "png": ".png",
        "jfif": ".jfif",
        "bmp": ".bmp",
        "tiff": ".tiff",
        "webp": ".webp",
    }

    # Define a extensão correta ou usa .jpg como padrão
    file_extension = extension_map.get(detected_type, ".jpg")

    # Cria caminho temporário para salvar a imagem recebida
    tmp_path = tempfile.mktemp(suffix=file_extension)

    try:
        # Salva o arquivo temporário
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Executa a busca de imagens semelhantes
        results = search_similar(tmp_path, top_k=top_k, threshold=threshold)

        return {"status": "success", "results": results}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar a imagem: {str(e)}"}

    finally:
        # Remove o arquivo temporário mesmo em caso de erro
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# rota para deletar imagens similares
@app.post("/delete-similar")
async def delete_similar(file: UploadFile = File(...), threshold: float = Form(...), top_k: int = Form(10)) -> JSONResponse:
    try:
        result = deletar_por_similaridade(file.filename, threshold, top_k)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
