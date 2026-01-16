import lancedb
import torch
import logging
from typing import List, Dict, Optional
from src.config import settings
from src.models import ModelRegistry

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        self.db = lancedb.connect(settings.DB_PATH)
        self.table = self.db.open_table("fashion_items")
        self.model, self.processor = ModelRegistry.get_embedding_model()

    def _embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length").to(settings.DEVICE)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def search(self, query: str, context_filter: Optional[str] = None, k: int = 5) -> List[Dict]:
        query_vector = self._embed_text(query)
        search_job = self.table.search(query_vector)
        
        if context_filter:
            search_job = search_job.where(f"caption LIKE '%{context_filter}%'", prefilter=True)
            
        results = search_job.limit(k).to_list()
        
        formatted_results = []
        for res in results:

            current_valid_path = settings.IMAGES_DIR / res["filename"]
            
            formatted_results.append({
                "filename": res["filename"],
                "score": 1 - res["_distance"],
                "caption": res["caption"],
                "path": str(current_valid_path) # <--- Force use of live path
            })
            
        return formatted_results