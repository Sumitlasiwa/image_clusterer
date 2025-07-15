"""
detector.py

Responsible for initializing the face detection and embedding model
from InsightFace. This module is called from main to process faces.
"""
import insightface

def load_face_model(ctx_id=0): 
    """
    Loads and prepares the InsightFace model.
    
    Args:
        ctx_id (int): GPU ID (0 = use GPU, -1 = use CPU)
    
    Returns:
        model: Loaded FaceAnalysis model
    """
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id = ctx_id)
    return model