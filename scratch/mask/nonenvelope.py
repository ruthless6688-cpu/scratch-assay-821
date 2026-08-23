from .maksing import Masking
import numpy as np

class MaskingNonEnvelop(Masking):
    """无包络版：仅方向性填充 + 孔洞填充 + 最大连通域，不做任何闭运算/填平"""
    def process(image: np.ndarray) -> np.ndarray:
        mask = Masking.overlay_extract(image)
        return Masking._finalize(mask)