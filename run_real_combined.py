# -*- coding: utf-8 -*-
"""真实划痕图：中轴法 vs 扫描线法 对比（双口径：包络版 + 无包络版）

对 Input/ 下每张图同时输出两种口径的结果：
  [包络版]   41px 椭圆闭运算填平边缘细胞凹陷（类包络掩膜）
  [无包络版] 仅孔洞填充 + 最大连通域（保留真实边缘形态）

输出文件命名（与历史版本一致）：
  包络版   _real{idx}_mask.png / _real{idx}_mask_ridge.png / _real{idx}_profile.png
  无包络版 _real{idx}_mask_nofill.png / _real{idx}_mask_nofill_ridge.png / _real{idx}_profile_nofill.png
"""
import sys, io, os, re, logging
from datetime import datetime
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from skimage.morphology import medial_axis
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 彻底杜绝 findfont 刷屏：把字体查找的日志级别提到 ERROR（只保留真正错误）
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ---- 路径：以脚本所在目录为锚点，整目录拷贝即可换机器运行 ----
PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR = PROJECT_ROOT / 'Input'            # 单张输入图像放这里
OUT_DIR = PROJECT_ROOT / 'Output_result'    # 输出结果统一放这里
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 中文字体：优先项目自带，其次跨平台候选探测，找不到则警告并回退 ----
CJK_FONT_CANDIDATES = [
    str(PROJECT_ROOT / 'fonts' / 'msyh.ttc'),   # 项目自带字体（随项目一起分发，跨机器渲染一致）
    r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyh.ttf',
    r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\simsun.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
_cjk_font = next((p for p in CJK_FONT_CANDIDATES if Path(p).exists()), None)
_cjk_family = None
if _cjk_font is not None:
    try:
        font_manager.fontManager.addfont(_cjk_font)
        _cjk_family = font_manager.FontProperties(fname=_cjk_font).get_name()
    except Exception:
        _cjk_font = None
if _cjk_family is not None:
    # 最终校验：用 findfont 实际查一次，查得到才设置（彻底杜绝 not found 警告）
    try:
        font_manager.findfont(font_manager.FontProperties(family=_cjk_family),
                              fallback_to_default=True)
        plt.rcParams['font.family'] = [_cjk_family, 'sans-serif']
    except Exception:
        _cjk_family = None
plt.rcParams['axes.unicode_minus'] = False
if _cjk_family is None:
    print('[提示] 未加载到中文字体，图表中文可能显示为方块；项目 fonts/ 目录存在时不会出现')

def load_cjk_font(size: int):
    """PIL 用中文字体；找不到时回退默认字体。"""
    if _cjk_font is not None:
        try:
            return ImageFont.truetype(_cjk_font, size)
        except Exception:
            pass
    return ImageFont.load_default()

from scratch_width_proto import (centerline_width, order_path, longest_path,
                                 arc_sample, scanline_width, stats, fmt,
                                 clean_mask_envelope, clean_mask_minimal)

# ================= 可调参数（按实验条件调整） =================
# 包络版闭运算结构元素直径（px）：决定"填平多宽的边缘细胞凹陷"。
# 经验起点 41px；应随物镜倍率/图像分辨率标定（高倍率/高分辨率需增大）。
ENVELOPE_KERNEL = 41

# ---- 自动读取 Input 目录下所有图片 ----
IMGS = sorted([p for p in IMG_DIR.iterdir()
               if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
if not IMGS:
    raise SystemExit(f'[错误] {IMG_DIR} 下没有图片，请把输入图像放入该目录')

# ---- 命令行模式：--single [序号|文件名关键字] 只处理一张；默认批量处理全部 ----
def select_images(argv):
    """按命令行参数返回待处理图片列表。
    --single            -> 只取第一张
    --single 3          -> 只取排序后第 3 张
    --single 131449     -> 只取文件名包含 131449 的图
    其他（无 --single）  -> 全部
    """
    if '--single' not in argv:
        return IMGS
    sel = argv[argv.index('--single') + 1] if len(argv) > argv.index('--single') + 1 else None
    if sel is None:
        return [IMGS[0]]
    if sel.isdigit() and len(sel) <= 2:          # 两位以内数字视为序号
        idx = int(sel)
        if 1 <= idx <= len(IMGS):
            return [IMGS[idx - 1]]
        raise SystemExit(f'[错误] 序号 {idx} 超出范围 1~{len(IMGS)}')
    hits = [p for p in IMGS if sel.lower() in p.name.lower()]
    if not hits:
        raise SystemExit(f'[错误] 找不到文件名包含 "{sel}" 的图片')
    return [hits[0]]


# ---------------- 工具：图像加载 / 掩膜提取 / 掩膜清理 ----------------
def load_rgb(path):
    im = Image.open(path).convert('RGB')
    return np.array(im)


def overlay_extract(rgb):
    """从分割叠加图中提取黄色标记区域作为伤口掩膜（不重新分割）"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    return ((h >= 15) & (h <= 40) & (s >= 60) & (v >= 40)).astype(np.uint8)


# ---------------- 单图双口径处理 ----------------
def process_image(idx, name, img, m, tag, label, out_dir):
    """对一张掩膜做测宽对比 + 出图；tag 用于文件名后缀，label 用于打印/标题"""
    out_dir.mkdir(parents=True, exist_ok=True)
    frac = m.mean()
    ys, xs = np.nonzero(m)
    hspan, wspan = ys.max() - ys.min(), xs.max() - xs.min()
    vertical = hspan > wspan
    print(f'[{label}] 掩膜像素占比: {frac*100:.1f}%   方向: {"竖直" if vertical else "水平"} (高{hspan} 宽{wspan})')

    # 中轴法
    dist, skel, (sx, sy, sw) = centerline_width(m)
    path = order_path(sx, sy)
    wmap = {(int(x), int(y)): w for x, y, w in zip(sx, sy, sw)}
    pos, ws = arc_sample(path, wmap, n=80)
    s_center_all = stats(ws)
    # 有效测量区：剔除两端收窄段（宽度 < 0.8×最大宽度的点不参与统计）
    wmax = np.nanmax(ws)
    keep = ws >= 0.8 * wmax
    s_center = stats(ws[keep]) if keep.sum() >= 3 else s_center_all

    # 扫描线法（竖直划痕 -> 逐行测宽，等价于转置）
    sw_ = scanline_width(m.T) if vertical else scanline_width(m)
    s_scan = stats(sw_)

    print(f'  [中轴中心线法]  全段: {fmt(s_center_all)}')
    print(f'  [中轴中心线法]  有效测量区(剔除两端): {fmt(s_center)}')
    print(f'  [扫描线法]      {fmt(s_scan)}')

    # ---- 出图 ----
    mask8 = (m * 255).astype(np.uint8)

    # 普通叠加图：伤口淡蓝色
    color_plain = img.copy()
    color_plain[mask8 > 0] = (0, 200, 255)
    Image.fromarray(color_plain).save(os.path.join(out_dir, f'_real{idx}_mask{tag}.png'))

    # 山脊版：绿=原始山脊(含毛刺)  红=清理后主干
    color = img.copy()
    color[mask8 > 0] = (0, 200, 255)
    raw_skel = medial_axis(m.astype(bool)).astype(np.uint8)
    rys, rxs = np.nonzero(raw_skel)
    for (px, py) in zip(rxs[::2], rys[::2]):
        cv2.circle(color, (int(px), int(py)), 1, (0, 255, 0), -1)
    trunk = longest_path(sx, sy)
    for (px, py) in trunk:
        cv2.circle(color, (int(px), int(py)), 2, (255, 0, 0), -1)
    pil_img = Image.fromarray(color)
    im_draw = ImageDraw.Draw(pil_img)
    font = load_cjk_font(22)
    im_draw.text((20, 20), '绿点=原始山脊(含毛刺)  红点=清理后主干', fill=(255, 255, 255),
                 stroke_width=2, stroke_fill=(0, 0, 0), font=font)
    pil_img.save(os.path.join(out_dir, f'_real{idx}_mask_ridge{tag}.png'))

    # 宽度分布对比图
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pos, ws, 'b-o', ms=3, lw=1, label='中轴中心线法（沿弧长）')
    ax.axhline(s_center['mean'], color='b', ls='--', lw=1,
               label=f"中轴均值(有效测量区) {s_center['mean']:.1f}")
    # 两端剔除区间用浅色底标出
    ax.axvspan(pos[0], pos[keep.argmax()], color='gray', alpha=0.15)
    ax.axvspan(pos[-1 - keep[::-1].argmax()], pos[-1], color='gray', alpha=0.15)
    xx = np.arange(len(sw_))
    ok = ~np.isnan(sw_)
    ax.plot(xx[ok], sw_[ok], 'r.', ms=2, alpha=0.5,
            label='扫描线法（逐' + ('行' if vertical else '列') + '范围）')
    ax.axhline(s_scan['mean'], color='r', ls='--', lw=1, label=f"扫描线均值 {s_scan['mean']:.1f}")
    ax.set_xlabel('位置（像素）'); ax.set_ylabel('宽度（像素）')
    ax.set_title(f'图{idx} 宽度对比（{label}）：中轴法 vs 扫描线法（{name[:20]}…）')
    ax.legend(); ax.grid(alpha=0.3)
    # 右下角说明：SD 与 CV
    info = (f"中轴法(有效测量区): 均值 {s_center['mean']:.1f}  SD {s_center['sd']:.1f}  "
            f"CV {s_center['cv']:.3f}\n"
            f"扫描线法:            均值 {s_scan['mean']:.1f}  SD {s_scan['sd']:.1f}  "
            f"CV {s_scan['cv']:.3f}")
    ax.text(0.98, -0.06, info, transform=ax.transAxes, ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#f5f8fb', ec='#b8c4d0'))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'_real{idx}_profile{tag}.png'), dpi=110)
    plt.close()


def image_tag(name):
    """取图片名标识：文件名中最后一段数字（拍摄时间戳，如 131516）；没有则取文件名主干前 12 字符"""
    nums = re.findall(r'\d+', os.path.splitext(name)[0])
    return nums[-1] if nums else os.path.splitext(name)[0][:12]


def main():
    todo = select_images(sys.argv[1:])
    print(f'[模式] 共 {len(todo)} 张待处理：' + ', '.join(p.name for p in todo))
    run_ts = datetime.now().strftime('%m%d_%H%M')      # 本次运行统一时间戳（月日_时分，无秒）
    for idx, path in enumerate(todo, 1):
        name = os.path.basename(path)
        img = load_rgb(path)
        print(f'\n========== 图 {idx}: {name}  ({img.shape[1]}x{img.shape[0]}) ==========')

        # 每次输出独立子文件夹：图片标识_时间戳
        out_dir = OUT_DIR / f'{image_tag(name)}_{run_ts}'
        print(f'输出目录: {out_dir}')

        raw = overlay_extract(img)

        # 包络版（41px 闭运算填平边缘凹陷）
        print('--- [包络版] 类包络：闭运算41px填平边缘细胞凹陷 ---')
        process_image(idx, name, img, clean_mask_envelope(raw, ENVELOPE_KERNEL), '', '包络版', out_dir)

        # 无包络版（保留真实边缘形态）
        print('--- [无包络版] 仅孔洞填充+最大连通域，保留真实边缘 ---')
        process_image(idx, name, img, clean_mask_minimal(raw), '_nofill', '无包络版', out_dir)

        print(f'已输出: {out_dir.name}/_real{idx}_mask.png / _real{idx}_mask_nofill.png / '
              f'_real{idx}_profile.png / _real{idx}_profile_nofill.png')


if __name__ == '__main__':
    main()
