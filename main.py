import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.routes import app as router 

app = FastAPI()

# 1. Incluir rotas do router
app.include_router(router)

# 2. Incluir as Imagens
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "mnt", "faroldigital-cloud", "media")

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# 3. Servir frontend 
FRONT_DIR = os.path.join(os.path.dirname(__file__), "front")

app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")