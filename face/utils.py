
"""
utils.py

Provides utility functions for handling faces:
- Drawing bounding boxes on images
- Cropping face regions and saving
"""

import cv2
from typing import List, Tuple

def draw_multiple_bboxes(image_path: str, bboxes: List[Tuple[int, int, int, int]]):
    """
    Draws bounding boxes on the given image and returns the image.
    
    Args:
        image_path: Path to the image
        bboxes: List of (x1, y1, x2, y2) bounding boxes
    
    Returns:
        image: Image with drawn boxes
    """
    image = cv2.imread(image_path)
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def crop_and_save_face(image_path: str, bbox: Tuple[int, int, int, int], save_path: str):
    """
    Crops a face region from an image and saves it.

    Args:
        image_path: Path to original image
        bbox: (x1, y1, x2, y2) coordinates of face
        save_path: Where to save cropped face
    """
    image = cv2.imread(image_path)
    if image is None: return

    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]

    if cropped.size > 0:
        cv2.imwrite(save_path, cropped)

def is_blurry(image, area , MIN_FACE_AREA, threshold=6.4):
    """
    Checks if an image (face crop) is blurry based on Laplacian variance.
    
    Args:
        image: Cropped face image to check.
        area: Area (w × h) of the detected face.
        MIN_FACE_AREA: Minimum area threshold to consider a valid face.
        threshold (float): images with value less than threshold are considered blurry
    return:
        True if the face is blurry or too small; otherwise, False.
    """
    
    if area < MIN_FACE_AREA:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold 