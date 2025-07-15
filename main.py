# """
# main.py

# This is the main pipeline runner.
# Steps:
# 1. Load model
# 2. Detect faces, extract embeddings
# 3. Store in ChromaDB
# 4. Cluster embeddings
# 5. Save results in folders with crops and metadata
# """

# from config import INPUT_FOLDER, CHROMA_DB_PATH, METADATA_CSV,IMAGE_FOLDER, CLUSTERED_OUTPUT, IMAGE_EXTENSIONS, BATCH_SIZE
# from face.detector import load_face_model
# from db_utils.chroma_manager import init_chroma, get_collection
# from clustering.cluster_faces import cluster_embeddings
# from io_utils.image_writer import save_cluster_images
# import cv2
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# from collections import defaultdict
# from more_itertools import chunked
# from pathlib import Path

# # def process_images():
# #     """
# #     Detect faces, extract embeddings, and store in ChromaDB.
    
# #     Returns:
# #         Chroma collection
# #     """
# #     model = load_face_model()
# #     chroma_client = init_chroma(str(CHROMA_DB_PATH))
# #     collection = get_collection(chroma_client)
    
# #     df = pd.DataFrame(columns=["image_path", "face_index", "bbox"])
# #     embeddings, ids, metadatas = [], [], []

# #     for file in tqdm(INPUT_FOLDER.rglob("*")):    # Search all files recursively
# #         if file.suffix.lower() not in IMAGE_EXTENSIONS:
# #             continue

# #         path = INPUT_FOLDER / file
# #         img = cv2.imread(path)
# #         faces = model.get(img)

# #         for idx, face in enumerate(faces):
# #             x1, y1, x2, y2 = face.bbox.astype(int)
# #             bbox_str = f"{x1},{y1},{x2},{y2}"
# #             face_id = f"{path.stem}_{idx}"

# #             df.loc[len(df)] = [path, idx, bbox_str]
# #             ids.append(face_id)
# #             embeddings.append(face.embedding.tolist())
# #             metadatas.append({"image_path": str(path), "bbox": bbox_str, "face_index": idx})
        
# #     df.to_csv(METADATA_CSV, index=False)
    
# #     for id_chunk, emb_chunk, meta_chunk in zip(chunked(ids, BATCH_SIZE), chunked(embeddings, BATCH_SIZE), chunked(metadatas, BATCH_SIZE)):
# #         collection.add(ids=id_chunk, embeddings=emb_chunk, metadatas=meta_chunk)
# #     return collection

# # def main():
# #     collection = process_images()
# #     data = collection.get(include=["embeddings", "metadatas"])

# #     embeddings = np.array(data["embeddings"])
# #     metadatas = data["metadatas"]

# #     for m in metadatas:
# #         m["bbox"] = tuple(map(int, m["bbox"].split(","))) # Convert bounding box "x1,y1,x2,y2" string to a tuple (x1, y1, x2, y2) for easier use.

# #     clusters = cluster_embeddings(embeddings)
  
# #     grouped = defaultdict(list)
# #     for cid, meta in zip(clusters, metadatas):
# #         if cid != -1:
# #             grouped[cid].append(meta)

# #     for cid, faces in grouped.items():
# #         save_cluster_images(cid, faces, str(CLUSTERED_OUTPUT))

# # if __name__ == "__main__":
# #     main()

# """
# main.py

# This is the main pipeline runner.
# Steps:
# 1. Load model
# 2. Detect faces, extract embeddings
# 3. Store in ChromaDB
# 4. Cluster embeddings
# 5. Save results in folders with crops and metadata
# """

# from config import INPUT_FOLDER, OUTPUT_FOLDER, CHROMA_DB_PATH, METADATA_CSV, IMAGE_FOLDER, CLUSTERED_OUTPUT
# from face.detector import load_face_model
# from db_utils.chroma_manager import init_chroma, get_collection
# from clustering.cluster_faces import cluster_embeddings
# from io_utils.image_writer import save_cluster_images
# import os, cv2
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# from collections import defaultdict

# def process_images():
#     """
#     Detect faces, extract embeddings, and store in ChromaDB.
    
#     Returns:
#         Chroma collection
#     """
#     model = load_face_model()
#     chroma_client = init_chroma(str(CHROMA_DB_PATH))
#     collection = get_collection(chroma_client)

#     os.makedirs(IMAGE_FOLDER, exist_ok=True)
#     df = pd.DataFrame(columns=["image_path", "face_index", "bbox"])
#     embeddings, ids, metadatas = [], [], []

#     for file in tqdm(os.listdir(INPUT_FOLDER)):
#         if not file.lower().endswith((".jpg", ".png", ".jpeg")):
#             continue

#         path = os.path.join(INPUT_FOLDER, file)
#         img = cv2.imread(path)
#         faces = model.get(img)

#         for idx, face in enumerate(faces):
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             bbox_str = f"{x1},{y1},{x2},{y2}"
#             face_id = f"{os.path.splitext(file)[0]}_{idx}"

#             df.loc[len(df)] = [path, idx, bbox_str]
#             ids.append(face_id)
#             embeddings.append(face.embedding.tolist())
#             metadatas.append({"image_path": path, "bbox": bbox_str, "face_index": idx})
        
#         cv2.imwrite(os.path.join(IMAGE_FOLDER, file), img)

#     df.to_csv(METADATA_CSV, index=False)
#     collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
#     return collection

# def main():
#     collection = process_images()
#     data = collection.get(include=["embeddings", "metadatas"])

#     embeddings = np.array(data["embeddings"])
#     metadatas = data["metadatas"]

#     for m in metadatas:
#         m["bbox"] = tuple(map(int, m["bbox"].split(",")))

#     clusters = cluster_embeddings(embeddings)

#     grouped = defaultdict(list)
#     for cid, meta in zip(clusters, metadatas):
#         if cid != -1:
#             grouped[cid].append(meta)

#     for cid, faces in grouped.items():
#         save_cluster_images(cid, faces, str(CLUSTERED_OUTPUT))

# if __name__ == "__main__":
#     main()

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
        if not file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            continue

        img = cv2.imread(str(file)) # OpenCV still needs string path
        faces = model.get(img)

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            bbox_str = f"{x1},{y1},{x2},{y2}"
            face_crop = img[y1:y2, x1:x2]
            if face_crop is None or face_crop.size == 0:
                continue
            w, h = x2 - x1, y2 - y1
            area = w * h
            if is_blurry(face_crop, area, MIN_FACE_AREA = 6500):
                print(f"[BLUR] Skipping blurry face in {file}")
                
                cv2.imwrite(str(BLUR_IMG / f"{Path(file).stem}_{idx}.jpg"), face_crop)

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
