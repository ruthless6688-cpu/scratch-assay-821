# -*- coding: utf-8 -*-
"""真实划痕图：宽度测量（原始版，数据输出版）

对 Input/ 下每张图计算原始版宽度统计：
  [原始版] 仅方向性填充 + 孔洞填充 + 最大连通域（保留真实边缘形态）
  （全局闭运算包络版已停用 2026-08-22；自适应半岛修正版暂停用 2026-08-22）

输出：每张图一个子文件夹（图片标识_月日_时分），内含 result.txt
     （掩膜占比、方向、中轴法全段/有效区、扫描线法的宽度统计）
"""
import sys, io, os, re
from datetime import datetime
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---- 路径：以脚本所在目录为锚点，整目录拷贝即可换机器运行 ----
PROJECT_ROOT = Path(__file__).resolve().parent
IMG_DIR = PROJECT_ROOT / 'Input'            # 单张输入图像放这里
OUT_DIR = PROJECT_ROOT / 'Output_result'    # 输出结果统一放这里
OUT_DIR.mkdir(parents=True, exist_ok=True)

from scratch_width_proto import (centerline_width, order_path, longest_path,
                                 arc_sample, scanline_width, stats, fmt,
                                 clean_mask_minimal)

# ---- 中文字体：优先项目自带，其次跨平台候选探测（用于 PIL 叠加图标注）----
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

def load_cjk_font(size: int):
    """PIL 用中文字体；找不到时回退默认字体。"""
    if _cjk_font is not None:
        try:
            return ImageFont.truetype(_cjk_font, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ================= 可调参数（按实验条件调整） =================
# [已停用] 全局闭运算包络核（2026-08-22 起不再使用）
# ENVELOPE_KERNEL = 12

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


# ---------------- 单图处理（原始版） ----------------
def process_image(idx, name, img, m, tag, label, out_dir):
    """对一张掩膜做测宽统计，返回数据字典。"""
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

    # ---- 输出 2 张图（叠加图 + 山脊图）----
    mask8 = (m * 255).astype(np.uint8)

    # 叠加图：伤口淡蓝色
    color_plain = img.copy()
    color_plain[mask8 > 0] = (0, 200, 255)
    Image.fromarray(color_plain).save(os.path.join(out_dir, f'_real{idx}_mask{tag}.png'))

    # 山脊图：绿=原始山脊(含毛刺)  红=清理后主干
    from skimage.morphology import medial_axis
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

    # ---- 返回统计数据（供 txt 汇总）----
    return {
        'label': label,
        'tag': tag,
        'mask_frac': float(frac),
        'vertical': vertical,
        'hspan': int(hspan), 'wspan': int(wspan),
        'center_all': s_center_all,
        'center_eff': s_center,
        'scanline': s_scan,
    }


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

        # 原始版（保留真实边缘形态）
        print('--- [原始版] 仅孔洞填充+最大连通域，保留真实边缘 ---')
        blue_mask = clean_mask_minimal(raw)
        d_min = process_image(idx, name, img, blue_mask, '_nofill', '原始版', out_dir)

        # [已停用] 自适应半岛修正版（2026-08-22）——front_line 方案暂不采用
        # 修正逻辑集中在 scratch_width_proto.correct_peninsulas，需要时重新启用
        d_cor = d_min   # 修正版暂等于原始版

        # 汇总写入 result.txt（原始版）
        txt_path = out_dir / 'result.txt'
        with open(txt_path, 'w', encoding='utf-8') as fh:
            fh.write(f'图片: {name}\n')
            fh.write(f'尺寸: {img.shape[1]}x{img.shape[0]}\n')
            fh.write(f'时间戳: {run_ts}\n\n')
            d = d_min
            fh.write('===== 原始版 =====\n')
            fh.write(f'掩膜像素占比: {d["mask_frac"]*100:.2f}%\n')
            fh.write(f'方向: {"竖直" if d["vertical"] else "水平"} (高{d["hspan"]} 宽{d["wspan"]})\n\n')
            for key, title in (('center_all', '中轴中心线法·全段'), ('center_eff', '中轴中心线法·有效测量区(剔除两端)'),
                               ('scanline', '扫描线法')):
                s = d[key]
                fh.write(f'[{title}] n={s["n"]} 均值={s["mean"]:.2f} 中位={s["median"]:.2f} '
                         f'SD={s["sd"]:.2f} CV={s["cv"]:.4f} P10={s["p10"]:.1f} P90={s["p90"]:.1f} '
                         f'min={s["mn"]:.1f} max={s["mx"]:.1f}\n')
        print(f'数据已写入: {txt_path}')


if __name__ == '__main__':
    main()
