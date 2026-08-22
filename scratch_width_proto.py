# -*- coding: utf-8 -*-
"""
划痕分析算法库（纯函数，无 IO / 绘图 / 自测入口）
模块一 掩膜清理：_largest_component / _directional_fill（方向性孔洞填充，解决端部挂边细胞团）
                  / _finalize / clean_mask_minimal（原始版，保留真实边缘）
                  / 自适应半岛修正：estimate_stable_core / axis_profile / correct_peninsulas
                  （全局闭运算 clean_mask_envelope 已停用，2026-08-22）
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

# [已停用] 全局闭运算包络版（2026-08-22 起不再使用，保留历史对照）
# 问题：固定核(41px)只看一个尺度，不知道细胞大小/划痕宽度/局部形态；
# 由自适应半岛修正 correct_peninsulas() 取代。
# def clean_mask_envelope(mask, kernel=41):
#     """包络版：椭圆闭运算填平边缘细胞凹陷 + 孔洞填充 + 最大连通域。
#     kernel 为闭运算结构元素直径（px），由调用方按物镜倍率标定传入。"""
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
#     m = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, k)
#     return _finalize(m)

def clean_mask_minimal(mask):
    """无包络版：仅方向性填充 + 孔洞填充 + 最大连通域，不做任何闭运算/填平"""
    return _finalize(mask)


# [已停用] 预处理：闭运算 + 孔洞填充（死代码，无任何调用，2026-08-22 注释）
# 原功能被 clean_mask_minimal / _finalize 取代
# def preprocess(mask, close_k=5):
#     k = np.ones((close_k, close_k), np.uint8)
#     m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
#     m = ndimage.binary_fill_holes(m).astype(np.uint8)
#     return m


# ---------------- 2. 欧氏距离变换 + 中轴中心线 ----------------
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


# ---------------- 3. 对比：等间距垂直扫描线法（每列纵向范围） ----------------
def scanline_width(mask):
    ys, xs = np.nonzero(mask)
    widths = np.full(mask.shape[1], np.nan)
    for x in range(mask.shape[1]):
        col = ys[xs == x]
        if len(col):
            widths[x] = col.max() - col.min() + 1
    return widths


# ---------------- 4. 统计 ----------------
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



# ---------------- 5. 自适应半岛修正（基于 mask 轮廓前沿线的局部自适应包络） ----------------
def correct_peninsulas(mask, smooth_win=15,
                       depth_ratio=0.15, mad_coef=4.0, cell_frac=0.05,
                       endpoint_frac=0.12):
    """自适应半岛修正（front_line 方案）：用 front_line.extract_front_line
    提取蓝色 mask 的左右切口前沿线，检测突入伤口的细胞团并选择性填充。

    基线取"外沿分位包络"（右线 90 分位 / 左线 10 分位，窗口约 4% 弧长）：
    半岛只会让前沿线向内偏移、不会进入外沿分位，因此包络不被半岛自身污染。

    返回 dict: corrected / correction / stats
    """
    m = np.asarray(mask, dtype=bool)
    # 用 front_line 提取左右前沿线
    try:
        from front_line import extract_front_line
        left_pts, right_pts = extract_front_line(m)
    except Exception:
        left_pts = right_pts = np.empty((0, 2))
    if len(left_pts) < 8 or len(right_pts) < 8:
        return {'corrected': m, 'correction': np.zeros_like(m),
                'stats': {'failed': True, 'correction_area_ratio': 0.0,
                          'max_correction_depth': 0.0, 'corrected_section_ratio': 0.0,
                          'review_flag': True, 'reason': 'front_line_failed'}}

    # 全局主轴（轮廓 PCA）→ 宽度方向 v
    main = _largest_component(m)
    yy, xx = np.nonzero(main)
    pxy = np.column_stack((xx.astype(np.float64), yy.astype(np.float64)))
    center = pxy.mean(axis=0)
    cov = np.cov(pxy - center, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
        u = vecs[:, int(np.argmax(vals))]
    except np.linalg.LinAlgError:
        return {'corrected': m, 'correction': np.zeros_like(m),
                'stats': {'failed': True, 'correction_area_ratio': 0.0,
                          'max_correction_depth': 0.0, 'corrected_section_ratio': 0.0,
                          'review_flag': True, 'reason': 'pca_failed'}}
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]], dtype=np.float64)

    # PCA 特征向量符号是任意的：强制约定"右线位于 n 较大的一侧"，
    # 否则 intrude 极性随图翻转（真半岛全部漏检、反而检出端部锥度）。
    if float(np.median((right_pts - center) @ v)) < float(np.median((left_pts - center) @ v)):
        v = -v

    def _side_deviation(pts, sign):
        """一条前沿线：投影到 (s, n)，等弧长重采样(1px)后提取外沿分位包络。
        sign=+1 右线(大 n 侧), -1 左线(小 n 侧)。"""
        rel = pts - center
        s = rel @ u
        n = rel @ v
        order = np.argsort(s)
        s, n = s[order], n[order]
        # 等弧长重采样到 1px，使后续窗口宽度/端点剔除都是像素单位
        s_u = np.arange(s[0], s[-1] + 1e-6, 1.0)
        n_u = np.interp(s_u, s, n)
        pts_u = np.column_stack([np.interp(s_u, s, pts[order, 0]),
                                 np.interp(s_u, s, pts[order, 1])])
        # 外沿分位包络：窗宽约 4% 弧长并随线长自适应（限幅 41~151px）。
        # 半岛向内偏移，不会进入外沿分位 → 包络不会被半岛自身污染。
        win = int(np.clip(round(0.04 * (s_u[-1] - s_u[0])), 41, 151))
        q = 90 if sign > 0 else 10
        env = ndimage.percentile_filter(n_u, q, size=win, mode='nearest')
        n_smooth = _local_median(env, smooth_win)
        n_end = max(1, int(round(len(s_u) * endpoint_frac)))
        # 突入：向伤口内部（中心线方向）偏移
        # 右线（大 n 侧）：突入 = n_smooth - n（n 变小=向中心）
        # 左线（小 n 侧）：突入 = n - n_smooth（n 变大=向中心）
        if sign > 0:
            intrude = np.maximum(0.0, n_smooth - n_u)
        else:
            intrude = np.maximum(0.0, n_u - n_smooth)
        intrude[:n_end] = 0
        intrude[len(s_u) - n_end:] = 0
        return {'s': s_u, 'n': n_u, 'n_smooth': n_smooth, 'intrude': intrude,
                'n_end': n_end, 'pts': pts_u}

    L = _side_deviation(left_pts, -1)
    R = _side_deviation(right_pts, +1)

    # 阈值（自适应）
    med_w = float(np.median(2.0 * cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5)[m]))
    cell_est = med_w * cell_frac
    l_mad = np.median(np.abs(L['intrude'][L['n_end']:-L['n_end']])) if len(L['intrude']) > 2 * L['n_end'] else 0.0
    r_mad = np.median(np.abs(R['intrude'][R['n_end']:-R['n_end']])) if len(R['intrude']) > 2 * R['n_end'] else 0.0
    left_thr = max(cell_est, med_w * depth_ratio, l_mad * mad_coef)
    right_thr = max(cell_est, med_w * depth_ratio, r_mad * mad_coef)

    left_pen = L['intrude'] > left_thr
    right_pen = R['intrude'] > right_thr

    # 填充：沿局部法线从原始前沿点填到平滑位置，只填原本是细胞的像素
    correction = np.zeros_like(m)
    h, w = m.shape

    def _fill(dev, pen, sign):
        pts = dev['pts']
        s_arr, n_arr = dev['s'], dev['n']
        n_smooth = dev['n_smooth']
        for i in np.nonzero(pen)[0]:
            p0 = pts[i]
            n0 = n_arr[i]
            n1 = n_smooth[i]
            lo, hi = min(n0, n1), max(n0, n1)
            # 沿宽度方向 v 从 n0 扫到 n1（用全局 v 近似法线）
            for k in np.arange(lo, hi + 1e-6, 1.0):
                p = p0 + v * (k - n0)
                xi, yi = int(round(p[0])), int(round(p[1]))
                if 0 <= xi < w and 0 <= yi < h and not m[yi, xi]:
                    correction[yi, xi] = 1

    _fill(L, left_pen, -1)
    _fill(R, right_pen, +1)

    correction = (correction > 0) & (~m)

    # [方案A] 形态学磨边：开运算去掉孤立毛刺 + 中值滤波平滑边界，
    # 避免逐截面"一刀切"造成的突兀台阶。
    if correction.any():
        c8 = (correction * 255).astype(np.uint8)
        c8 = cv2.morphologyEx(c8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        c8 = cv2.medianBlur(c8, 3)
        correction = (c8 > 127) & (~m)

    corrected = m | correction

    area_ratio = float(correction.sum()) / max(float(m.sum()), 1.0)
    depths = np.concatenate([L['intrude'][left_pen], R['intrude'][right_pen]])
    max_depth = float(depths.max()) if depths.size else 0.0
    n_pen = int(left_pen.sum()) + int(right_pen.sum())
    section_ratio = n_pen / max(len(left_pen) + len(right_pen), 1)
    review_flag = (area_ratio > 0.15) or (section_ratio > 0.4)

    return {
        'corrected': corrected,
        'correction': correction,
        'stats': {
            'failed': False,
            'correction_area_ratio': area_ratio,
            'max_correction_depth': max_depth,
            'corrected_section_ratio': section_ratio,
            'peninsula_sections': n_pen,
            'left_threshold': left_thr, 'right_threshold': right_thr,
            'review_flag': review_flag,
        },
    }


def _local_median(x, win=15):
    """滑动中位数（鲁棒平滑），边界用 nearest 填充。"""
    if win % 2 == 0:
        win += 1
    if win < 3 or len(x) < win:
        return x.astype(np.float64)
    return ndimage.median_filter(x.astype(np.float64), size=win, mode='nearest')
