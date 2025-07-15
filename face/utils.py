# """
# utils.py

# Provides utility functions for handling faces:
# - Drawing bounding boxes on images
# - Cropping face regions and saving
# """

# import cv2
# from typing import List, Tuple

# def draw_bbox(image_path: str, bbox: List[Tuple[int, int, int, int]]):
#     """
#     Draws bounding box on the detected face of image and returns the image.
#     Args:
#         image_path: Path to the image
#         bboxes: List of (x1,y1,x2,y2) bounding boxes
        
#     Returns:
#         image: Image with drawn box
#     """
#     image = cv2.imread(image_path) #, cv2.IMREAD_REDUCED_COLOR_2) # cv2.IMREAD_REDUCED_COLOR_2 will reduce resolution of image to half
#     x1,y1,x2,y2 = bbox
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     return image

# def crop_and_save(image_path: str, bbox: Tuple[int, int, int, int], save_path: str):
#     """
#     Crops a face region from an image and saves it.

#     Args:
#         image_path (str): Path to original image
#         bbox : (x1, y1, x2, y2) coordinates of face
#         save_path (str): Where to save cropped face
#     """
#     image = cv2.imread(image_path) #, cv2.IMREAD_REDUCED_COLOR_2) # cv2.IMREAD_REDUCED_COLOR_2 will reduce resolution of image to half
#     if image is None: return
    
#     x1,y1,x2,y2 = bbox
#     cropped = image[y1:y2, x1:x2]
    
#     if cropped.size > 0:
#         cv2.imwrite(save_path, cropped)
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
    checks if image is blurry or not up to the threshold
    
    Args:
        image: image to check for blurriness
        threshold (float, optional): images with value less than threshold are considered blurry
    return:
        True if blurry
    """
    

    if area < MIN_FACE_AREA:
        print(f"[SKIP] Small/partial face detected in {image}")
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold 