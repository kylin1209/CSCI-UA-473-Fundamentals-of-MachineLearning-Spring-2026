"""
Process TMDB Dataset: Download from HuggingFace and generate embeddings.

This script:
1. Downloads TMDB 5000 Movies dataset from HuggingFace
2. Generates text embeddings using Nomic AI model
3. Saves processed data to data/processed/tmdb_embedded.parquet

Usage:
    python process_data.py

About Dataset:
    - 5,000 movies with titles, descriptions, genres, ratings, revenue
    - Source: https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies
    - Size: ~10MB
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_NAME = "AiresPucrs/tmdb-5000-movies"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "tmdb_embedded.parquet"
EMBEDDING_DIM = 768

# Columns to keep from the dataset
KEEP_COLUMNS = [
    "id",
    "title",
    "overview",
    "genres",
    "vote_average",
    "vote_count",
    "revenue",
    "runtime",
    "release_date",
    "poster_path",
    "backdrop_path",
]


def download_dataset():
    """Download TMDB dataset from HuggingFace."""
    print(f"📥 Downloading {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        print(f"✅ Downloaded {len(dataset)} movies")
        return dataset
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        sys.exit(1)


def prepare_data(dataset):
    """Convert HuggingFace dataset to pandas DataFrame with cleaned data."""
    print("🔧 Preparing data...")
    
    # Convert to pandas
    df = dataset.to_pandas()
    
    # Print available columns for debugging
    print(f"   Available columns: {list(df.columns)}")
    
    # Select and rename columns (only keep those that exist)
    available_cols = [col for col in KEEP_COLUMNS if col in df.columns]
    df = df[available_cols].copy()
    
    # Filter out rows with missing overviews (we can't embed empty text)
    df = df[df["overview"].notna() & (df["overview"].str.len() > 0)].copy()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Add local_poster_path (for image display in Streamlit) if poster_path exists
    if "poster_path" in df.columns:
        df["local_poster_path"] = df["poster_path"].apply(
            lambda x: f"data/posters/{Path(x).name}" if pd.notna(x) and x else None
        )
    else:
        df["local_poster_path"] = None
    
    print(f"✅ Prepared {len(df)} movies with valid descriptions")
    return df


def load_model():
    """Load the embedding model."""
    print(f"🧠 Loading model {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print(f"✅ Model loaded ({EMBEDDING_DIM} dimensions)")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


def generate_embeddings(df, model, batch_size=32):
    """Generate embeddings for all movie overviews."""
    print(f"🔤 Generating embeddings for {len(df)} movies...")
    print(f"   (Batch size: {batch_size}, this may take 2-3 minutes)")
    
    texts = df["overview"].tolist()
    embeddings = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    return embeddings


def save_data(df, embeddings):
    """Save dataframe with embeddings to parquet."""
    print(f"💾 Saving to {OUTPUT_FILE}...")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add embeddings as a column
    df["embedding"] = [emb for emb in embeddings]
    
    # Save to parquet
    try:
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df)} movies with embeddings")
        print(f"   File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        sys.exit(1)


def main():
    """Main pipeline."""
    print("=" * 70)
    print("TMDB 5000 Movies: Download & Embed")
    print("=" * 70)
    
    # Check if already processed
    if OUTPUT_FILE.exists():
        print(f"⚠️  Output file already exists: {OUTPUT_FILE}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("Aborting.")
            return
    
    # Pipeline
    dataset = download_dataset()
    df = prepare_data(dataset)
    model = load_model()
    embeddings = generate_embeddings(df, model)
    save_data(df, embeddings)
    
    print("=" * 70)
    print("✅ Done! Your dataset is ready for Lab 3.")
    print("=" * 70)


if __name__ == "__main__":
    main()
