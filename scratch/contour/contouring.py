from abc import ABC, abstractmethod
import numpy as np

class Contouring(ABC):
    """轮廓线提取算法抽象接口

    基于二值掩码图像，提取目标的左前沿、右前沿与中心线，
    接收 Masking 算法输出的 mask 作为输入。
    """

    @abstractmethod
    def process(mask: np.ndarray, image: np.ndarray, draw_debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """从二值掩码中提取左前沿、右前沿、中心线，可选输出调试可视化图像

        Parameters
        ----------
        mask : np.ndarray
            二值掩码图像，由Masking.process生成；
            非0为目标有效区域，0为背景屏蔽区域
        image : np.ndarray
            原始输入图像，仅用于调试图像绘制
        draw_debug : bool, default=False
            是否生成调试可视化图像。为True时返回绘制完成的调试图；
            为False时，调试图像返回空数组。

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            返回元组 (left_leading_edge, right_leading_edge, center_line, debug_image)

            left_leading_edge : np.ndarray
                左前沿点集，N×2数组，每一行代表一个点 (x, y)
            right_leading_edge : np.ndarray
                右前沿点集，N×2数组，每一行代表一个点 (x, y)
            center_line : np.ndarray
                中心线点集，N×2数组，每一行代表一个点 (x, y)
            debug_image : np.ndarray
                调试可视化图像；draw_debug=True时为叠加轮廓的结果图；
                draw_debug=False时返回空数组。
        """
        pass
