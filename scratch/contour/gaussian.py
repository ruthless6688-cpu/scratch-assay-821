from .contouring import Contouring
from scipy.ndimage import gaussian_filter1d
import numpy as np

class ContouringGuassian(Contouring):
    """高斯插值法"""

    def process(mask: np.ndarray, image: np.ndarray = None, draw_debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """利用高斯插值法获取前沿线和中心线"""
        left_x, right_x = ContouringGuassian.get_row_boundaries(mask)
        left_s, right_s = ContouringGuassian.smooth_boundaries(left_x, right_x, sigma=10)
        center_x = ContouringGuassian.build_centerline(left_s, right_s)

        return left_x, right_x, center_x, []

    def fill_nan_1d(arr):
        """
        线性插值填充 NaN
        """
        x = np.arange(len(arr))
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return arr.copy()

        filled = np.interp(x, x[valid], arr[valid])
        return filled


    def get_row_boundaries(mask):
        """
        对每一行提取最左和最右前景像素
        """
        H, W = mask.shape
        left_x = np.full(H, np.nan, dtype=np.float32)
        right_x = np.full(H, np.nan, dtype=np.float32)

        for y in range(H):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) > 0:
                left_x[y] = xs[0]
                right_x[y] = xs[-1]

        return left_x, right_x

    def smooth_boundaries(left_x, right_x, sigma=8):
        """
        插值 + 高斯平滑
        """
        left_f = ContouringGuassian.fill_nan_1d(left_x)
        right_f = ContouringGuassian.fill_nan_1d(right_x)

        left_s = gaussian_filter1d(left_f, sigma=sigma)
        right_s = gaussian_filter1d(right_f, sigma=sigma)

        return left_s, right_s

    def build_centerline(left_s, right_s):
        """
        中心线取左右平滑边界中点
        """
        center_x = (left_s + right_s) / 2.0
        return center_x

    def refine_centerline_to_inside(mask, center_x):
        """
        保证中心线始终在 mask 内：
        如果某一行 center_x 落在前景外，则投影回该行前景中点
        """
        H, W = mask.shape
        refined = center_x.copy()

        for y in range(H):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) == 0:
                continue

            c = int(round(refined[y]))
            if c < xs[0]:
                refined[y] = xs[0]
            elif c > xs[-1]:
                refined[y] = xs[-1]
            elif mask[y, c] == 0:
                # 如果落在孔洞/空白里，回到这一行前景区间中心
                refined[y] = (xs[0] + xs[-1]) / 2.0

        return refined
