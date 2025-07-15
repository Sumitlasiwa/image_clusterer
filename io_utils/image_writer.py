
"""
image_writer.py

Handles:
- Creating folders for each cluster
- Cropping face images
- Drawing boxes on originals
- Writing cluster_images.json mapping
"""

import os, cv2
import json
from face.utils import crop_and_save_face, draw_multiple_bboxes
from pathlib import Path

def create_cluster_dirs(base_dir: Path, cluster_id: int):
    """
    Create directory structure for a given cluster.
    - crop/ : holds one representative face
    - originals/ : all annotated source images

    Returns:
        cluster_dir path
    """
    cluster_dir = base_dir / f"cluster_{cluster_id}"
    (cluster_dir / "crop").mkdir(parents=True, exist_ok=True)
    (cluster_dir / "originals").mkdir(parents=True, exist_ok=True)
    return cluster_dir

def save_cluster_images(cluster_id, faces, base_dir):
    """
    Saves a face crop and annotated originals for a given cluster.

    Args:
        cluster_id: The cluster label
        faces: List of metadata dictionaries
        base_dir: Output directory
    """
    cluster_dir = create_cluster_dirs(base_dir, cluster_id)
    crop_path = cluster_dir / "crop" / "face.jpg"
    
    # Save representative face crop
    crop_and_save_face(faces[0]["image_path"], faces[0]["bbox"], crop_path)

    # Group by image
    from collections import defaultdict
    image_to_bboxes = defaultdict(list)
    for face in faces:
        image_to_bboxes[face["image_path"]].append(face["bbox"])

    # Annotate and save each image
    cluster_dict = {}
    for i, (img_path, bboxes) in enumerate(image_to_bboxes.items()):
        image = draw_multiple_bboxes(img_path, bboxes)
        save_path = cluster_dir / "originals" / Path(img_path).name   # Path(img_path).name gives just the filename (e.g: "image_01.jpg")
        cluster_dict[str(i)] = img_path
        cv2.imwrite(save_path, image)

    # Save JSON mapping
    with open(os.path.join(cluster_dir, "originals", "cluster_images.json"), "w") as f:
        json.dump(cluster_dict, f, indent=4)
