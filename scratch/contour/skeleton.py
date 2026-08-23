from .contouring import Contouring
import numpy as np

class ContouringSkeleton(Contouring):
    """骨架法"""
    
    def process(mask: np.ndarray, image: np.ndarray, draw_debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass