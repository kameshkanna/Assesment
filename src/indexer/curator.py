import json
import logging
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.config import settings
from src.models import ModelRegistry

logger = logging.getLogger(__name__)

class DataCurator:
    """
    Handles the 'Feature Extraction' phase by enriching raw images 
    with dense semantic captions using Florence-2.
    """
    def __init__(self):
        self.model, self.processor = ModelRegistry.get_caption_model()

    def process_directory(self, source_dir: Path, output_file: Path):
        """
        Iterates through images, generates captions, and saves metadata.
        Supports resume capability via line counting.
        """
        source_dir = Path(source_dir)
        image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        
        # Check existing progress
        processed_files = set()
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        processed_files.add(entry['filename'])
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Found {len(image_files)} images. {len(processed_files)} already processed.")
        
        # Processing Loop
        batch_images = []
        batch_filenames = []
        
        with open(output_file, 'a') as f_out:
            for img_path in tqdm(image_files, desc="Curating Metadata"):
                if img_path.name in processed_files:
                    continue
                
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                    batch_filenames.append(img_path.name)
                    
                    if len(batch_images) >= settings.BATCH_SIZE:
                        self._flush_batch(batch_images, batch_filenames, f_out)
                        batch_images, batch_filenames = [], []
                        
                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {e}")

            # Flush remaining
            if batch_images:
                self._flush_batch(batch_images, batch_filenames, f_out)

    def _flush_batch(self, images, filenames, file_handle):
        """Internal method to run inference and write to disk."""
        task_prompt = "<MORE_DETAILED_CAPTION>"
        prompts = [task_prompt] * len(images)
        
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(settings.DEVICE, settings.DTYPE)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, text in enumerate(generated_texts):
            # Parse result
            parsed = self.processor.post_process_generation(
                text, 
                task=task_prompt, 
                image_size=(images[i].width, images[i].height)
            )
            caption = parsed[task_prompt].replace("<pad>", "").strip()
            
            record = {
                "filename": filenames[i],
                "caption": caption,
                "path": str(settings.IMAGES_DIR / filenames[i])
            }
            file_handle.write(json.dumps(record) + "\n")
            file_handle.flush()