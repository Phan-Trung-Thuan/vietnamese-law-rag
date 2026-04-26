# Vietnamese Law RAG System

This project is a Retrieval-Augmented Generation (RAG) system specialized for Vietnamese Law, using a fine-tuned Llama2-7B model, BM25, and semantic search.

## Repository Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: The original Jupyter notebooks used for research and development.
- `src/`: The main source code, structured by functionality:
  - `data/crawler.py`: Logic for scraping Vietnamese law Q&A.
  - `retrieval/retriever.py`: Implements indexing and retrieval mechanisms using BM25.
  - `models/train.py`: Fine-tunes the base language model.
  - `models/inference.py`: Inference script for generating answers.
  - `pipeline/rag_pipeline.py`: The end-to-end RAG workflow combining retrieval and generation.
- `requirements.txt`: Python dependencies needed to run the project.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Each module under `src/` can be executed independently or imported. Ensure you have properly configured the data paths and downloaded any necessary model weights before running the pipeline.
