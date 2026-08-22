import cv2
import numpy as np
from scipy import ndimage

def extract_mask_contour(mask):
    """提取 imageMask 最大连通域的完整外轮廓，返回 (x, y) 点。"""
    binary = np.asarray(mask, dtype=np.uint8)
    labels, count = ndimage.label(binary > 0, structure=np.ones((3, 3)))
    if count == 0:
        return np.empty((0, 2), dtype=float)
    sizes = ndimage.sum(binary > 0, labels, range(1, count + 1))
    component = (labels == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=float)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def remove_top_bottom_edges(contour, image_height):
    """只处理图像上下边界：边界线保留首尾点，其他轮廓点全部保留。"""
    if len(contour) == 0:
        return contour.copy()
    trimmed = contour.copy()
    top_boundary = np.isclose(contour[:, 1], 0)
    bottom_boundary = np.isclose(contour[:, 1], image_height - 1)
    for boundary in (top_boundary, bottom_boundary):
        indices = np.flatnonzero(boundary)
        if len(indices) > 2:
            trimmed[indices[1:-1]] = np.nan
    return trimmed


def _walk_to_bottom(contour, start, step, image_height):
    """从一个上边界点沿轮廓顺序走到下边界，保留真实轮廓点。"""
    points = []
    index = start
    visited = set()
    while index not in visited:
        visited.add(index)
        point = contour[index]
        if np.isfinite(point).all():
            points.append(point)
            if point[1] >= image_height - 1:
                break
        index = (index + step) % len(contour)
    return np.asarray(points, dtype=float)


def split_left_right_fronts(trimmed_contour, image_height):
    """从上边界首尾点沿轮廓延伸，并按起点 x 坐标命名左右前沿。"""
    if len(trimmed_contour) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    finite = np.isfinite(trimmed_contour).all(axis=1)
    top_indices = np.flatnonzero(finite & (trimmed_contour[:, 1] == 0))
    if len(top_indices) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    path_a = _walk_to_bottom(trimmed_contour, int(top_indices[0]), 1, image_height)
    path_b = _walk_to_bottom(trimmed_contour, int(top_indices[-1]), -1, image_height)
    if len(path_a) == 0 or len(path_b) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # 轮廓顺时针/逆时针均可；只依据两个起点的图像 x 坐标确定左右。
    if path_a[0, 0] <= path_b[0, 0]:
        return path_a, path_b
    return path_b, path_a

def extract_front_line(mask):
    """提取前沿线，输入处理后的mask，返回左右前沿线"""
    contour = extract_mask_contour(mask)
    trimmed_contour = remove_top_bottom_edges(contour, mask.shape[0])
    return split_left_right_fronts(trimmed_contour, mask.shape[0])