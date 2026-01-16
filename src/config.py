import os
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # --- DYNAMIC ROOT DETECTION ---
    # Logic: This file is in evals/src/config.py -> parent.parent = evals/
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    
    # --- DATA PATHS ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Images Folder
    IMAGES_DIR: Path = DATA_DIR / "images"
    
    # Database Folder (LanceDB creates this)
    DB_PATH: Path = DATA_DIR / "lancedb_index"
    
    # Metadata File
    METADATA_PATH: Path = DATA_DIR / "metadata_captions_clean.jsonl"
    
    # --- HARDWARE ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # --- MODELS ---
    EMBEDDING_MODEL_ID: str = "google/siglip-so400m-patch14-384"
    CAPTION_MODEL_ID: str = "microsoft/Florence-2-large"
    
    # --- TUNING ---
    BATCH_SIZE: int = 64

    @staticmethod
    def setup():
        """Creates data folders if they don't exist."""
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Auto-run setup
settings = Config()
settings.setup()