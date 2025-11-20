import torch
import clip
import numpy as np
from PIL import Image

class CLIPService:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"CLIP carregado: {model_name} ({self.device})")

    def get_embedding(self, img_path: str) -> np.ndarray:
        image = self.preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return embedding.cpu().numpy().flatten()
