from abc import ABC, abstractmethod
import numpy as np
import cv2
from scipy import ndimage

class Masking(ABC):
    """遮罩处理算法抽象接口

    根据输入图像生成二值掩码(mask)，用于后续轮廓提取 contouring 处理。
    输出掩码图像中，有效区域像素为非0，屏蔽区域像素为0。
    """

    @abstractmethod
    def process(image: np.ndarray, data) -> np.ndarray:
        """执行遮罩处理，由输入图像输出二值掩码

        Parameters
        ----------
        image : np.ndarray
            输入原始图像/待处理图像，支持灰度图或BGR图像

        Returns
        -------
        np.ndarray
            生成的二值掩码 mask，掩码图像与输入图像尺寸保持一致；
            非0像素代表有效目标区域，0像素代表被屏蔽区域，供后续轮廓线提取使用。
        """
        pass

    def overlay_extract(rgb):
        """从分割叠加图中提取黄色标记区域作为伤口掩膜（不重新分割）"""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.int32)
        v = hsv[:, :, 2].astype(np.int32)
        return ((h >= 15) & (h <= 40) & (s >= 60) & (v >= 40)).astype(np.uint8)

    def _largest_component(mask):
        """返回二值图的最大 8 连通域。"""
        m = np.asarray(mask, dtype=bool)
        lbl, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            return np.zeros_like(m, dtype=bool)
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        return lbl == (int(np.argmax(sizes)) + 1)

    def _directional_fill(mask):
        """按划痕自身主轴做方向性孔洞填充。

        目的：普通 binary_fill_holes 会把图像四周都当成外部背景入口，导致
        划痕端部连到图像边界的细胞团无法填充。这里不使用固定的上下方向，
        而是先估计划痕主轴，再把背景连通性限制为只能从划痕两侧进入：
        - 与两侧相连的背景/细胞保留；
        - 只通过划痕两端进入、但不与两侧相连的内部细胞填充。
        """
        src = np.asarray(mask, dtype=bool)
        if src.sum() < 16:
            return src

        # 先用最大连通域确定主伤口，避免少量噪声影响主轴估计。
        main = Masking._largest_component(src)
        yy, xx = np.nonzero(main)
        if len(xx) < 16:
            return src

        # PCA 估计划痕长度方向 u；v 为垂直于划痕的宽度方向。
        pts = np.column_stack((xx.astype(np.float64), yy.astype(np.float64)))
        center = pts.mean(axis=0)
        cov = np.cov(pts - center, rowvar=False)
        try:
            vals, vecs = np.linalg.eigh(cov)
            u = vecs[:, int(np.argmax(vals))]
        except np.linalg.LinAlgError:
            return src
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.array([-u[1], u[0]], dtype=np.float64)

        # 全图像素投影到划痕坐标 (s: 长度方向, n: 宽度方向)。
        h, w = src.shape
        gy, gx = np.indices((h, w), dtype=np.float64)
        px = gx - center[0]
        py = gy - center[1]
        s_all = px * u[0] + py * u[1]
        n_all = px * v[0] + py * v[1]
        s_main = s_all[main]
        n_main = n_all[main]

        # 沿主轴建立每个截面的左右边界包络。
        s0 = float(s_main.min())
        s_idx = np.floor(s_main - s0).astype(np.int32)
        n_bins = int(s_idx.max()) + 1
        if n_bins < 8:
            return src
        left = np.full(n_bins, np.inf, dtype=np.float64)
        right = np.full(n_bins, -np.inf, dtype=np.float64)
        np.minimum.at(left, s_idx, n_main)
        np.maximum.at(right, s_idx, n_main)
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() < max(4, n_bins // 4):
            return src
        axis = np.arange(n_bins, dtype=np.float64)
        left = np.interp(axis, axis[valid], left[valid])
        right = np.interp(axis, axis[valid], right[valid])

        # 边界轻度平滑，避免单个毛刺把端部走廊拉得过宽。
        smooth_size = max(3, int(round(n_bins / 120)) * 2 + 1)
        if smooth_size >= 5:
            left = ndimage.median_filter(left, size=smooth_size, mode='nearest')
            right = ndimage.median_filter(right, size=smooth_size, mode='nearest')
        # [防御] 平滑后复查：独立中值滤波可能让个别 bin 右界 <= 左界，
        # 不复查会带着坏包络继续算，静默产出错误填充结果。
        if np.any(right <= left):
            return src

        s_grid = np.floor(s_all - s0).astype(np.int32)
        s_grid = np.clip(s_grid, 0, n_bins - 1)
        left_grid = left[s_grid]
        right_grid = right[s_grid]
        width0 = float(np.median(right - left))
        if not np.isfinite(width0) or width0 < 4:
            return src

        # 只在伤口包络及其窄边缘内工作，避免触碰两侧完整细胞层。
        side_margin = max(2.0, min(8.0, 0.04 * width0))
        inside = (n_all >= left_grid) & (n_all <= right_grid)
        work_region = (n_all >= left_grid - side_margin) & (n_all <= right_grid + side_margin)
        bg = (~main) & work_region

        # 在划痕自身坐标中封住两端，只允许背景从左右两侧进入。
        # 端部封帽厚度很小，仅用于阻断"从图像端部绕进来的连通路径"。
        cap = max(2, int(round(0.015 * n_bins)))
        endpoint = (s_grid <= cap) | (s_grid >= (n_bins - 1 - cap))
        bg_for_search = bg & ~endpoint

        side_band = (
            ((n_all >= left_grid - side_margin) & (n_all < left_grid)) |
            ((n_all > right_grid) & (n_all <= right_grid + side_margin))
        )
        seeds = bg_for_search & side_band
        if not np.any(seeds):
            # 没有可靠侧面种子时，退回保守的普通孔洞填充。
            return ndimage.binary_fill_holes(main)

        # 找到与左右侧连通的背景；其余包络内背景视为内部细胞并填充。
        lbl, n_lbl = ndimage.label(bg_for_search, structure=np.ones((3, 3)))
        seed_labels = np.unique(lbl[seeds])
        seed_labels = seed_labels[seed_labels != 0]
        reachable = np.isin(lbl, seed_labels)
        internal_cells = inside & bg & ~reachable
        repaired = main | internal_cells
        return repaired


    def _finalize(mask):
        """方向性内部细胞填充 + 孔洞填充 + 最大连通域（两种口径共用）。"""
        # 不再只依赖 binary_fill_holes；先按划痕主轴填充端部/内部细胞。
        m = Masking._directional_fill(mask > 0)
        m = ndimage.binary_fill_holes(m).astype(np.uint8)
        m = Masking._largest_component(m).astype(np.uint8)
        return m

