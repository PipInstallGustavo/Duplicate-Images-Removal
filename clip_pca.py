#1. Importar libs
import os
import torch
import clip
import numpy as np
import joblib 
from PIL import Image
from sklearn.decomposition import PCA
from pymilvus import connections, list_collections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict

# Variáveis globais
collection = None
pca_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # carregar modelo do CLIP


#2. Configurar o Milvus
def config_db_vetorial(collection_name:str, dim_pca:int=128) -> Collection:
    # conectar
    connections.connect(alias="default", host="localhost", port="19530")

    # Criar collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim_pca),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields, description="Image similarity search with CLIP embeddings")

    # Remover collections existentes, se houver
    if collection_name in list_collections():
        Collection(collection_name).drop()
    
    # Criar collection
    collection = Collection(name=collection_name, schema=schema)

    # Criar índice
    collection.create_index("embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    })

    return collection


#4. Função para extrair embedding CLIP
def get_clip_embedding(img_path:str) -> np.array:
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy().flatten()


#5. Indexar imagens com PCA
def index_images(folder:str, batch_size:int=200):
    embeddings = []
    paths = []
    
    print("Processando imagens para indexação...")
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
            path = os.path.join(folder, file)
            emb = get_clip_embedding(path)
            embeddings.append(emb)
            paths.append(path)
            print(f"Imagem processada: {file}")

    embeddings = np.array(embeddings)
    dim_pca = 128
    
    # Treinar PCA
    pca_model = PCA(n_components=dim_pca)
    embeddings_reduced = pca_model.fit_transform(embeddings)

    # caminho para salvar o modelo do PCA
    pca_model_path = "pca_model.joblib"
    
    # Salvar PCA para uso futuro
    joblib.dump(pca_model, pca_model_path)
    print(f"PCA salvo em: {pca_model_path}")

    # Criar collection no Milvus
    collection = config_db_vetorial("image_clip")
    
    # Inserir em batch
    for i in range(0, len(paths), batch_size):
        batch_emb = embeddings_reduced[i:i+batch_size].tolist()
        batch_paths = paths[i:i+batch_size]
        collection.insert([batch_emb, batch_paths])

    collection.flush()
    collection.load()
    return collection, pca_model


# 6. Inicializar ou carregar Collection já existente 
def initialize_collection(collection_name:str):
    # garantir conexão antes de qualquer operação
    connections.connect(alias="default", host="localhost", port="19530")

    if collection_name in list_collections():
        collection = Collection(collection_name)

        pca_model_path = "pca_model.joblib"  # caminho para salvar o modelo do PCA
        
        # carregar modelo PCA já salvo, se houver
        if os.path.exists(pca_model_path):
            pca_model = joblib.load(pca_model_path)
            print(f"PCA carregado de: {pca_model_path}")
        else:
            print("Modelo PCA não configurado. Indexando imagens.")
            return None, None
        
        # Checar se as collections estão vazias
        if collection.num_entities == 0:
            print("Collections vazias. Reindexando imagens.")
            return None, None
        
        collection.load()
        return collection, pca_model
    else:
        print("Collection não encontrada. Indexando imagens...")
        return index_images("mnt/faroldigital-cloud/media")


# Inicializar coleção e modelo PCA (sem FastAPI, direto ao rodar o script)
collection, pca_model = initialize_collection("image_clip")

if collection is None or pca_model is None:
    print("Inicializando uma nova collection")
    collection, pca_model = index_images("mnt/faroldigital-cloud/media")


#7. Buscar imagens similares
def search_similar(img_path:str, threshold:float, top_k:int) -> List:
    emb = get_clip_embedding(img_path)
    emb_reduced = pca_model.transform([emb])[0]

    img_info = []

    results = collection.search(
        data=[emb_reduced.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["path"]
    )

    print(f"\nImagens similares a: {img_path}")
    for hit in results[0]:
        path_hit = hit.entity.get("path")
        filename = os.path.basename(path_hit)
        public_url = f"/media/{filename}"
        score = hit.score
        
        print(f"Cosine similarity: {score:.2f} | Path: {path_hit}")
        
        if score >= threshold:
            img_info.append((public_url, round(score, 2)))
    
    img_info.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(img_info)} images meeting threshold {threshold}")
    return img_info

# 8. deletar imagens

def deletar_por_similaridade(filename: str, threshold: float, top_k: int = 100) -> Dict:
    try:
        results_query = collection.query(
            expr=f'path LIKE "%{filename}"',
            output_fields=["path"]
        )
        
        if not results_query:
            return {
                "status": "success",
                "message": "Imagem de busca não encontrada no banco de dados.",
                "deleted_paths": []
            }
        
        query_path = results_query[0]["path"]

        emb = get_clip_embedding(query_path)
        emb_reduced = pca_model.transform([emb])[0]

        results = collection.search(
            data=[emb_reduced.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["path", "id"]
        )

        deleted_paths = []

        if results and results[0]:
            for hit in results[0]:
                hit_id = hit.entity.get("id")
                hit_path = hit.entity.get("path")
                hit_score = hit.score

                if hit_score >= threshold and hit_path != query_path:
                    print(f"Deletando {hit_path} (ID: {hit_id}) com similaridade: {hit_score:.2f}")
                    collection.delete(expr=f"id == {hit_id}")
                    deleted_paths.append(hit_path)

            collection.flush()
        
        return {
            "status": "success",
            "message": f"{len(deleted_paths)} imagens similares deletadas.",
            "deleted_paths": deleted_paths
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro: {str(e)}",
            "deleted_paths": []
        }

# 9. Função para adicionar novas imagens na collection sem indexar as imagens todas de novo
def add_new_images(folder:str):
    existing_paths = set()
    
    # Pegar os paths das imagens na collection
    results = collection.query(
        expr="",
        output_fields=["path"],
        limit=100000000
    )
    
    for item in results:
        existing_paths.add(item["path"])
    
    new_embeddings = []
    new_paths = []
    
    print("Checando por novas imagens")
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".jfif")):
            path = os.path.join(folder, file)
            if path not in existing_paths:
                emb = get_clip_embedding(path)
                new_embeddings.append(emb)
                new_paths.append(path)
                print(f"Nova imagem encontrada: {file}")
    
    if new_embeddings:
        new_embeddings = np.array(new_embeddings)
        new_embeddings_reduced = pca_model.transform(new_embeddings)
        
        # Inserir novas imagens
        collection.insert([new_embeddings_reduced.tolist(), new_paths])
        collection.flush()
        print(f"{len(new_paths)} novas imagens adicionadas à collection")
    else:
        print("Sem novas Imagens")
