"""
main.py

This is the main pipeline runner.
Steps:
1. Load model
2. Detect faces, extract embeddings
3. Store in ChromaDB
4. Cluster embeddings
5. Save results in folders with crops and metadata
"""

from config import INPUT_FOLDER, CHROMA_DB_PATH, METADATA_CSV, CLUSTERED_OUTPUT, BLUR_IMG, embeddings_file, metadata_file, init_dirs
from face.detector import load_face_model
from db_utils.chroma_manager import init_chroma, get_collection
from clustering.cluster_faces import cluster_embeddings
from io_utils.image_writer import save_cluster_images
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from face.utils import is_blurry


def process_images():
    """
    Detect faces, extract embeddings, and store in ChromaDB.
    
    Returns:
        Chroma collection
    """
    model = load_face_model()
    chroma_client = init_chroma(str(CHROMA_DB_PATH))
    collection = get_collection(chroma_client)

    # os.makedirs(IMAGES_FOLDER, exist_ok=True)
    df = pd.DataFrame(columns=["image_path", "face_index", "bbox"])
    embeddings, ids, metadatas = [], [], []
    

    for file in tqdm((INPUT_FOLDER.iterdir())):
        if file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        img = cv2.imread(str(file)) # OpenCV still needs string path
        faces = model.get(img)

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            bbox_str = f"{x1},{y1},{x2},{y2}"
            
            # Validate bbox
            if x1 >= x2 or y1 >= y2:
                continue
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop is None or face_crop.size == 0:
                continue
            
            area = x2 - x1 * y2 - y1
            if is_blurry(face_crop, area, MIN_FACE_AREA = 6500):
                print(f"[BLUR] Skipping blurry face in {file.name}")
                
                cv2.imwrite(str(BLUR_IMG / f"{file.stem}_{idx}.jpg"), face_crop)

                continue
            
            face_id = f"{Path(file).stem}_{idx}"

            df.loc[len(df)] = [file, idx, bbox_str]
            ids.append(face_id)
            embeddings.append(face.embedding.tolist())
            metadatas.append({"image_path": str(file), "bbox": bbox_str, "face_index": idx})
        
        # cv2.imwrite(os.path.join(IMAGES_FOLDER, file), img)

    df.to_csv(METADATA_CSV, index=False)
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    return collection

def main():
    init_dirs()             #ensure all important output folders defined in config.py are created before they’re used
    
    if not embeddings_file.exists() or not metadata_file.exists():
        print("[INFO] Running detection and saving cache...")
        collection = process_images()
        data = collection.get(include=["embeddings", "metadatas"])

        embeddings = data["embeddings"]
        metadatas = data["metadatas"]

        # Save embeddings
        np.save(embeddings_file, embeddings)

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadatas, f)

    else:
        print("[INFO] Loading from existing cache...")

        # Load from cache
        embeddings = np.load(embeddings_file)
        with open(metadata_file) as f:
            metadatas = json.load(f)
            
    for m in metadatas:
        m["bbox"] = tuple(map(int, m["bbox"].split(",")))

    clusters = cluster_embeddings(embeddings)

    grouped = defaultdict(list)
    for cid, meta in zip(clusters, metadatas):
        if cid != -1:
            grouped[cid].append(meta)

    for cid, faces in grouped.items():
        save_cluster_images(cid, faces, CLUSTERED_OUTPUT)

if __name__ == "__main__":
    main()
