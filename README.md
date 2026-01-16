# Fashion Retrieval Engine

A  implementation of **Hybrid Search** for fashion e-commerce. This project addresses the context blindness limitations of standard contrastive models by combining **Google SigLIP** (for high-fidelity visual embeddings) with **Microsoft Florence-2** (for automated metadata curation).

## Quick Start

The project is designed to run interactively via the Jupyter Notebook.

### 1. Prerequisites

Ensure a GPU-enabled environment is active.

⚠️ **Important**: Before running the search engine, you must unzip the `images.rar` file to the `data/images/` directory:

```bash
# Extract images from the RAR file
unrar x images.rar data/images/
# Or use 7-Zip if unrar is not available
7z x images.rar -odata/images/
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Execution Guide

Open `main.ipynb` and execute the cells in the following order:

- **Run Cell 1 (Setup)**: Imports dependencies and loads the configuration.

- **⚠️ SKIP Cell 2 (The Indexer)**:
  - Do not run the cell labeled "Build Index" or "Data Curator".
  - The Vector Database has been pre-computed and stored in `data/lancedb_index`. Running this cell will attempt to re-process the raw images and overwrite the existing index.

- **Run Cell 3 (Engine Initialization)**: Loads the Search Engine and models into memory.

- **Run Cell 4 (Engine Initialization and Dashboard)**: Loads the Search Engine and models into memory.
Launches the interactive UI.
  - Enter a query (e.g., `Red dress | Park`) and run the cell to visualize results.

## Project Structure

```
Evals/
├── main.ipynb                 # Interactive Dashboard (Entry Point)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   ├── images/                # Raw image repository
│   ├── metadata_captions_clean.jsonl  # Pre-computed metadata
│   └── lancedb_index/         # Pre-computed Vector Database
└── src/
    ├── indexer/               # Florence-2 Inference & Vector Construction
    │   ├── curator.py         # Metadata curation logic
    │   └── vector_db.py       # Vector database construction
    ├── retriever/             # Hybrid Search Logic (Vector + SQL Filtering)
    │   └── engine.py          # Search engine implementation
    ├── models.py              # Model initialization and management
    └── config.py              # Centralized configuration
```

## Methodology

The system utilizes an Enrich-then-Retrieve architecture to guarantee precision for compositional queries:

### Ingestion
Raw images are processed by **Florence-2-Large** to generate dense descriptive metadata (e.g., distinguishing "Studio" from "Street" environments).

### Indexing
**SigLIP-SO400M** generates normalized visual embeddings, stored in the pre-computed LanceDB vector database.

### Retrieval
A two-stage hybrid filter is applied:

1. **Visual Similarity**: Identifies garments that visually match the query.
2. **Contextual Filter**: Strictly rejects images that do not satisfy the environmental constraints defined in the metadata.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision-Language Model | Microsoft Florence-2-Large |
| Embedding Model | Google SigLIP-SO400M |
| Vector Database | LanceDB |
| Framework | PyTorch / Hugging Face Transformers |

## Key Features

- **Hybrid Search**: Combines visual similarity with contextual metadata filtering
- **Compositional Queries**: Supports complex queries with environment and style constraints
- **Pre-computed Index**: Optimized for quick inference without re-indexing
- **Interactive Dashboard**: User-friendly Jupyter-based interface

## Requirements

- CUDA-capable GPU (for model inference)
- Python 3.8+
- See `requirements.txt` for all dependencies

## Notes

- The vector database index is pre-computed and should not be regenerated unless you have new images to process.
- All model weights are downloaded on first run and cached locally.

