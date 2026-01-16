import lancedb
import logging
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.config import settings
from src.models import ModelRegistry

logger = logging.getLogger(__name__)

class VectorIndex:
    """
    Manages the LanceDB connection and the embedding pipeline (SigLIP).
    """
    def __init__(self):
        self.db = lancedb.connect(settings.DB_PATH)
        self.model, self.processor = ModelRegistry.get_embedding_model()
        self.table_name = "fashion_items"

    def _get_embeddings(self, images):
        """Generates normalized L2 embeddings."""
        inputs = self.processor(images=images, return_tensors="pt", padding="max_length").to(settings.DEVICE)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    def build_index(self, metadata_path: Path):
        """
        Reads enriched metadata, embeds images, and ingests into LanceDB.
        """
        if self.table_name in self.db.table_names():
            logger.warning(f"Dropping existing table: {self.table_name}")
            self.db.drop_table(self.table_name)

        def data_generator():
            batch_data = []
            batch_imgs = []
            
            with open(metadata_path, 'r') as f:
                lines = f.readlines()
                
            for line in tqdm(lines, desc="Indexing Vectors"):
                try:
                    record = json.loads(line)
                    img_path = Path(record["path"])
                    
                    if not img_path.exists():
                        continue
                        
                    img = Image.open(img_path).convert("RGB")
                    batch_data.append(record)
                    batch_imgs.append(img)
                    
                    if len(batch_imgs) >= settings.BATCH_SIZE:
                        vectors = self._get_embeddings(batch_imgs)
                        yield [dict(item, vector=v) for item, v in zip(batch_data, vectors)]
                        batch_data, batch_imgs = [], []
                        
                except Exception as e:
                    logger.error(f"Indexing error: {e}")
            
            if batch_imgs:
                vectors = self._get_embeddings(batch_imgs)
                yield [dict(item, vector=v) for item, v in zip(batch_data, vectors)]

        # Create Table
        logger.info("Starting LanceDB Ingestion...")
        tbl = self.db.create_table(self.table_name, data=data_generator())
        
        # Create FTS Index for Context Filtering
        logger.info("Creating Full-Text Search Index on captions...")
        tbl.create_fts_index("caption")
        logger.info(f"Index successfully built with {len(tbl)} records.")