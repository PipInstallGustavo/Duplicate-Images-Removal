from .clip_service import CLIPService
from .pca_service import PCAService

class DeleteService:
    def __init__(self, collection, pca_service: PCAService):
        self.collection = collection
        self.clip_service = CLIPService()
        self.pca_service = pca_service

    def delete_similar(self, filename: str, threshold: float, top_k: int = 100):
        try:
            results_query = self.collection.query(expr=f'path LIKE "%{filename}"', output_fields=["path"])
            if not results_query:
                return {"status": "success", "message": "Imagem não encontrada.", "deleted": []}

            query_path = results_query[0]["path"]
            emb = self.clip_service.get_embedding(query_path)
            emb_reduced = self.pca_service.transform([emb])[0]

            results = self.collection.search(
                data=[emb_reduced.tolist()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["path", "id"]
            )

            deleted = []
            for hit in results[0]:
                score = hit.score
                hit_id = hit.entity.get("id")
                hit_path = hit.entity.get("path")

                if score >= threshold and hit_path != query_path:
                    print(f"Deletando {hit_path} (similaridade {score:.2f})")
                    self.collection.delete(expr=f"id == {hit_id}")
                    deleted.append(hit_path)

            self.collection.flush()
            return {"status": "success", "deleted": deleted, "message": f"{len(deleted)} imagens removidas."}

        except Exception as e:
            return {"status": "error", "message": str(e), "deleted": []}
