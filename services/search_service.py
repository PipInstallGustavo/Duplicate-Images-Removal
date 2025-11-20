import os
from typing import List
from .clip_service import CLIPService
from .pca_service import PCAService

class SearchService:
    def __init__(self, collection, pca_service: PCAService):
        self.collection = collection
        self.clip_service = CLIPService()
        self.pca_service = pca_service

    def search(self, img_path: str, threshold: float, top_k: int = 5) -> List:
        emb = self.clip_service.get_embedding(img_path)
        emb_reduced = self.pca_service.transform([emb])[0]

        results = self.collection.search(
            data=[emb_reduced.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["path"]
        )

        img_info = []
        print(f"\nBuscando similares a: {img_path}")
        for hit in results[0]:
            path_hit = hit.entity.get("path")
            filename = os.path.basename(path_hit)
            public_url = f"/media/{filename}"
            score = hit.score

            if score >= threshold:
                img_info.append((public_url, round(score, 2)))

        img_info.sort(key=lambda x: x[1], reverse=True)
        print(f"{len(img_info)} resultados acima do threshold {threshold}.")
        return img_info
