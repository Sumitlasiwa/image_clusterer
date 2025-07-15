# # """
# # image_writer.py

# # Handles:
# # - Creating folders for each cluster
# # - Cropping face images
# # - Drawing boxes on originals
# # - Writing cluster_images.json mapping
# # """

# import os 
# import json
# import cv2
# from collections import defaultdict
# from face.utils import crop_and_save, draw_bbox

# # def create_cluster_dirs(base_dir: str, cluster_id: int):
# #     """
# #     Create directory sturcture for a given cluster.
# #     - crop/ : holds one representative face (face from first image in cluster)
# #     - originals/ : all annotated source images (only annotated on face of respective cluter)

# #     Returns:
# #         cluster_dir path
# #     """
# #     cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
# #     os.makedirs(os.path.join(cluster_dir, "crop"), exist_ok=True)
# #     os.makedirs(os.path.join(cluster_dir, "originals"), exist_ok=True)
# #     return cluster_dir

# # def save_cluster_images(cluster_id, faces, base_dir):
# #     """
# #     Saves a face crop and originals for a given cluster.
# #     maintains original image references in JSON for avoiding duplicacy later.

# #     Args:
# #         cluster_id: The cluster label
# #         faces: List of metadata dictionaries
# #         base_dir: Output directory
# #     """
# #     cluster_dir = create_cluster_dirs(base_dir, cluster_id)
# #     crop_path = os.path.join(cluster_dir, "crop", "face.jpg")
    
# #     # Save representative face crop (first face from clutered images)
# #     crop_and_save(faces[0]["image_path"], faces[0]["bbox"], crop_path)
    
# #     # Group by image

# #     cluster_dict = {}
 
    
# #     for i, face in enumerate(faces):
# #         image_path = face["image_path"]
# #         bbox = face["bbox"]
# #         image = draw_bbox(image_path, bbox)
# #         save_path = os.path.join(cluster_dir, "originals", os.path.basename(image_path))
# #         cluster_dict[str(i)] = image_path
# #         cv2.imwrite(save_path, image)
        
# #     # Save JSON mapping
# #     with open(os.path.join(cluster_dir, "originals", "cluster_images.json"), "w") as f:
# #         json.dump(cluster_dict, f, indent=4) 
# """
# image_writer.py

# Handles:
# - Creating folders for each cluster
# - Cropping face images
# - Drawing boxes on originals
# - Writing cluster_images.json mapping
# """

# # import os, cv2
# # import json
# # from collections import defaultdict
# # from face.utils import crop_and_save_face, draw_multiple_bboxes

# def create_cluster_dirs(base_dir: str, cluster_id: int):
#     """
#     Create directory structure for a given cluster.
#     - crop/ : holds one representative face
#     - originals/ : all annotated source images

#     Returns:
#         cluster_dir path
#     """
#     cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
#     os.makedirs(os.path.join(cluster_dir, "crop"), exist_ok=True)
#     os.makedirs(os.path.join(cluster_dir, "originals"), exist_ok=True)
#     return cluster_dir

# def save_cluster_images(cluster_id, faces, base_dir):
#     """
#     Saves a face crop and annotated originals for a given cluster.

#     Args:
#         cluster_id: The cluster label
#         faces: List of metadata dictionaries
#         base_dir: Output directory
#     """
#     cluster_dir = create_cluster_dirs(base_dir, cluster_id)
#     crop_path = os.path.join(cluster_dir, "crop", "face.jpg")
    
#     # Save representative face crop
#     crop_and_save(faces[0]["image_path"], faces[0]["bbox"], crop_path)

#     # Group by image
#     from collections import defaultdict
#     image_to_bboxes = defaultdict(list)
#     # for face in faces:
#     #     image_to_bboxes[face["image_path"]].append(face["bbox"])

#     # Annotate and save each image
#     cluster_dict = {}
#     for i, face in enumerate(faces):
#         image_path = face["image_path"]
#         bbox = face["bbox"]
#         image = draw_bbox(image_path, bbox)
#         save_path = os.path.join(cluster_dir, "originals", os.path.basename(image_path))
#         cluster_dict[str(i)] = image_path
#         cv2.imwrite(save_path, image)

#     # Save JSON mapping
#     with open(os.path.join(cluster_dir, "originals", "cluster_images.json"), "w") as f:
#         json.dump(cluster_dict, f, indent=4)
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
from collections import defaultdict
from face.utils import crop_and_save_face, draw_multiple_bboxes

def create_cluster_dirs(base_dir: str, cluster_id: int):
    """
    Create directory structure for a given cluster.
    - crop/ : holds one representative face
    - originals/ : all annotated source images

    Returns:
        cluster_dir path
    """
    cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
    os.makedirs(os.path.join(cluster_dir, "crop"), exist_ok=True)
    os.makedirs(os.path.join(cluster_dir, "originals"), exist_ok=True)
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
    crop_path = os.path.join(cluster_dir, "crop", "face.jpg")
    #hello world
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
        save_path = os.path.join(cluster_dir, "originals", os.path.basename(img_path))
        cluster_dict[str(i)] = img_path
        cv2.imwrite(save_path, image)

    # Save JSON mapping
    with open(os.path.join(cluster_dir, "originals", "cluster_images.json"), "w") as f:
        json.dump(cluster_dict, f, indent=4)
