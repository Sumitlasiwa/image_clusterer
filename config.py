# """
# config.py

# This file defines global paths and settings used across the project.
# Centralizing these makes it easier to maintain or change paths later.
# """

# from pathlib import Path

# INPUT_FOLDER = Path("data").resolve() # Folder where input images are located
# OUTPUT_FOLDER = Path("output").resolve() # Folder to store all output files
# CHROMA_DB_PATH = OUTPUT_FOLDER / "chroma_db" # ChromaDB directory path
# METADATA_CSV = OUTPUT_FOLDER / "metadata.csv" # Save path for metadata CSV
# IMAGE_FOLDER = OUTPUT_FOLDER / "images"
# CLUSTERED_OUTPUT = OUTPUT_FOLDER / "clustered_output" # Final clustered output folder (faces + annotations)

# IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
# DBSCAN_EPS = 0.5
# DBSCAN_MIN_SAMPLES = 2
# BATCH_SIZE = 200    # if you're dealing with thousands of faces, batch inserting in chunks reduces memory pressure and latency
"""
config.py

This file defines global paths and settings used across the project.
Centralizing these makes it easier to maintain or change paths later.
"""

from pathlib import Path

# Folder where input images are located
INPUT_FOLDER = Path("data").resolve()

# Folder to store all output files
OUTPUT_FOLDER = Path("output").resolve()

# ChromaDB directory path
CHROMA_DB_PATH = OUTPUT_FOLDER / "chroma_db"

# Save path for metadata CSV
METADATA_CSV = OUTPUT_FOLDER / "metadata.csv"

# Save path for visualized face-detected images
IMAGES_FOLDER = OUTPUT_FOLDER / "images"

# Final clustered output folder (faces + annotations)
CLUSTERED_OUTPUT = OUTPUT_FOLDER / "clustered_output"
