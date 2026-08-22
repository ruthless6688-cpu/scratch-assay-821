# -*- coding: utf-8 -*-
"""
划痕分析算法库（纯函数，无 IO / 绘图 / 自测入口）
模块一 掩膜清理：_largest_component / _directional_fill（方向性孔洞填充，解决端部挂边细胞团）
                  / _finalize / clean_mask_envelope（包络版闭运算）/ clean_mask_minimal（无包络版）
模块二 宽度测量：欧氏距离变换 + 中轴中心线法 vs 等间距扫描线法
输入：任意二值划痕掩膜（1=划痕/伤口，0=细胞）
由 run_real_combined.py 以真实图像驱动调用。
"""
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import medial_axis

# ---------------- 1. 掩膜清理 ----------------
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
    main = _largest_component(src)
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
    m = _directional_fill(mask > 0)
    m = ndimage.binary_fill_holes(m).astype(np.uint8)
    m = _largest_component(m).astype(np.uint8)
    return m


def clean_mask_envelope(mask, kernel=41):
    """包络版：椭圆闭运算填平边缘细胞凹陷 + 孔洞填充 + 最大连通域。
    kernel 为闭运算结构元素直径（px），由调用方按物镜倍率标定传入。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, k)
    return _finalize(m)


def clean_mask_minimal(mask):
    """无包络版：仅方向性填充 + 孔洞填充 + 最大连通域，不做任何闭运算/填平"""
    return _finalize(mask)


# ---------------- 2. 预处理：闭运算 + 孔洞填充 ----------------
def preprocess(mask, close_k=5):
    k = np.ones((close_k, close_k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    m = ndimage.binary_fill_holes(m).astype(np.uint8)
    return m


# ---------------- 3. 欧氏距离变换 + 中轴中心线 ----------------
def centerline_width(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)          # 距离场
    skel = medial_axis(mask.astype(bool)).astype(np.uint8)      # 中轴（scikit-image）
    skel = prune_centerline(skel, dist, frac=0.3)              # 主干筛选：按宽度阈值去毛刺
    ys, xs = np.nonzero(skel)
    w = 2.0 * dist[ys, xs]                                       # 局部宽度 = 2×距离
    return dist, skel, (xs, ys, w)


def prune_centerline(skel, dist, frac=0.55):
    """主干筛选：毛刺都贴着边缘（宽度小），主干宽度接近最大值——
    只保留宽度 >= frac×最大宽度 的骨架像素，再取最大连通分量，即可得到干净单线主干"""
    w = 2.0 * dist
    wmax = float(w[skel > 0].max())
    keep = (skel > 0) & (w >= frac * wmax)
    s = keep.astype(np.uint8)
    lbl, n = ndimage.label(s, structure=np.ones((3, 3)))
    if n > 0:
        sizes = ndimage.sum(s, lbl, range(1, n + 1))
        s = (lbl == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    return s


def order_path(xs, ys):
    """把骨架散点排成有序路径；从两个端点分别走线，取更长的一条（14:16 版）"""
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return np.zeros((0, 2), int)

    def walk(start):
        path, rem = [start], pts - {start}
        while rem:
            cand = {(path[-1][0] + dx, path[-1][1] + dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx, dy) != (0, 0)} & rem
            if not cand:
                break
            nxt = sorted(cand)[0]
            path.append(nxt)
            rem.discard(nxt)
        return np.array(path)

    def neigh(p):
        return {(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0)} & pts

    ends = [p for p in pts if len(neigh(p)) <= 1]
    if not ends:
        ends = [next(iter(pts))]
    best = max((walk(e) for e in ends), key=lambda p: len(p))
    return best


def longest_path(xs, ys):
    """两次 BFS 求骨架图的最长简单路径（真正主干：贯穿全程、无分叉）——用于掩膜图上画红线"""
    from collections import deque
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return np.zeros((0, 2), int)

    def bfs(start):
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        farthest = start
        while q:
            p = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (p[0] + dx, p[1] + dy)
                    if nb in pts and nb not in dist:
                        dist[nb] = dist[p] + 1
                        parent[nb] = p
                        q.append(nb)
                        if dist[nb] > dist[farthest]:
                            farthest = nb
        return farthest, parent

    a, _ = bfs(next(iter(pts)))
    b, parent = bfs(a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return np.array(path)


def arc_sample(path, wmap, n=60):
    """沿路径等弧长采样；wmap: {(x,y): 宽度}，返回 (弧长位置, 采样宽度)"""
    if len(path) < 2:
        return np.array([]), np.array([])
    d = np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return np.array([]), np.array([])
    pos = np.linspace(0, total, n)
    idx = np.clip(np.searchsorted(cum, pos), 0, len(path) - 1)
    ws = np.array([wmap.get((int(path[i][0]), int(path[i][1])), np.nan) for i in idx])
    return pos, ws


# ---------------- 4. 对比：等间距垂直扫描线法（每列纵向范围） ----------------
def scanline_width(mask):
    ys, xs = np.nonzero(mask)
    widths = np.full(mask.shape[1], np.nan)
    for x in range(mask.shape[1]):
        col = ys[xs == x]
        if len(col):
            widths[x] = col.max() - col.min() + 1
    return widths


# ---------------- 5. 统计 ----------------
def stats(w):
    w = w[~np.isnan(w)]
    return dict(n=len(w), mean=float(w.mean()), median=float(np.median(w)),
                sd=float(w.std()), cv=float(w.std() / w.mean()),
                p10=float(np.percentile(w, 10)), p90=float(np.percentile(w, 90)),
                mn=float(w.min()), mx=float(w.max()))


def fmt(s):
    return (f"n={s['n']}  均值={s['mean']:.2f}  中位={s['median']:.2f}  "
            f"标准差={s['sd']:.2f}  CV={s['cv']:.3f}  P10={s['p10']:.1f}  "
            f"P90={s['p90']:.1f}  min={s['mn']:.1f}  max={s['mx']:.1f}")


