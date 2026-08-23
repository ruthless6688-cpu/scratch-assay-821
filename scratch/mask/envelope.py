from .maksing import Masking
import numpy as np
import cv2

class MaskingEnvelop(Masking):
    """包络版：椭圆闭运算填平边缘细胞凹陷 + 孔洞填充 + 最大连通域。
    kernel 为闭运算结构元素直径（px），由调用方按物镜倍率标定传入。"""

    kernel = 41

    def process(image: np.ndarray) -> np.ndarray:
        mask = Masking.overlay_extract(image)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MaskingEnvelop.kernel, MaskingEnvelop.kernel))
        m = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, k)
        return Masking._finalize(m)
    
