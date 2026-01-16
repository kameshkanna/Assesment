import logging
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM
from src.config import settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Lazy-loading registry to manage heavy VLM and Embedding models.
    """
    _embed_model = None
    _embed_processor = None
    _caption_model = None
    _caption_processor = None

    @classmethod
    def get_embedding_model(cls):
        if cls._embed_model is None:
            logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL_ID}")
            cls._embed_model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL_ID).to(settings.DEVICE).eval()
            cls._embed_processor = AutoProcessor.from_pretrained(settings.EMBEDDING_MODEL_ID)
        return cls._embed_model, cls._embed_processor

    @classmethod
    def get_caption_model(cls):
        if cls._caption_model is None:
            logger.info(f"Loading Caption Model: {settings.CAPTION_MODEL_ID}")
            cls._caption_model = AutoModelForCausalLM.from_pretrained(
                settings.CAPTION_MODEL_ID, 
                trust_remote_code=True, 
                torch_dtype=settings.DTYPE
            ).to(settings.DEVICE).eval()
            cls._caption_processor = AutoProcessor.from_pretrained(
                settings.CAPTION_MODEL_ID, 
                trust_remote_code=True
            )
        return cls._caption_model, cls._caption_processor