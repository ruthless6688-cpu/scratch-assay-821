# 细胞划痕实验分析

## 项目背景

细胞划痕实验（scratch assay）通过在细胞单层上制造一条空白划痕，连续采集显微图像，观察细胞向划痕区域迁移并闭合的过程。该实验常用于评价细胞迁移能力、伤口修复能力以及药物对细胞运动的影响。

本项目是一个 Python 原型算法库，面向已经完成采集和基本预处理的 RGB 图像，提供以下分析流程：

1. 从图像中提取划痕区域掩膜。
2. 通过高斯平滑方法提取每一行的左右边界。
3. 计算划痕面积、平均宽度、宽度波动、边界粗糙度和细胞覆盖率。
4. 对连续图像进行逐帧分析，计算愈合率和表观迁移速度。
5. 可选地输入对照组图像，计算对照参考下的侵入数据和校正愈合率。

当前算法默认输入图像已经完成必要的配准；控制器内部不会执行帧间配准。

## 环境与依赖

建议使用 Python 3.10 或更高版本。主要依赖包括：

- NumPy：数组和数值计算
- OpenCV：图像颜色空间转换和形态学操作
- SciPy：孔洞填充、连通域处理和高斯滤波
- Pillow：测试代码中的图像读取
- pytest：运行测试

如果本地尚未安装依赖，可以执行：

```bash
pip install numpy opencv-python scipy pillow pytest
```

## 项目文件结构

```text
.
├─data/
│  ├─input/
│  │  ├─s1/                         # 实验组样例图像
│  │  └─s2/                         # 其他样例图像
│  └─output/                        # 分析输出目录，当前未固定输出格式
├─document/
│  ├─概要设计文档.adoc               # 算法概要设计和数学公式
│  ├─任务描述.pdf                    # 项目任务说明
│  ├─逻辑与量化指标梳理.pdf            # 指标和处理逻辑说明
│  └─量化分析算法大纲.pdf              # 量化分析设计资料
├─scratch/
│  ├─__init__.py                    # 暴露 DetectController
│  ├─detectcontroller.py            # 单帧和时序分析控制器
│  ├─result.py                      # 参数、结果和质量枚举定义
│  ├─mask/
│  │  ├─maksing.py                  # 掩膜基类和公共后处理
│  │  ├─nonenvelope.py              # 无包络掩膜处理
│  │  └─envelope.py                 # 包络形态学处理
│  └─contour/
│     ├─contouring.py               # 轮廓算法抽象接口
│     ├─gaussian.py                 # 高斯插值边界提取
│     └─skeleton.py                 # 骨架法接口，尚未实现
├─test/
│  ├─test_input.py                  # 输入相关测试
│  └─test_scratch.py                # 控制器测试和样例加载
├─pytest.ini
└─research.ipynb                    # 研究和算法验证笔记本
```

## 已完成的功能

### 掩膜处理

`MaskingNonEnvelop.process` 会从 RGB 图像中提取黄色标记区域作为划痕掩膜，并进行：

- 最大八连通域保留
- 孔洞填充
- 沿划痕主轴的方向性内部区域填充

`MaskingEnvelop` 额外提供椭圆核闭运算，用于平滑划痕边缘。当前控制器默认使用 `MaskingNonEnvelop`。

### 单帧分析

入口：

```python
from scratch import DetectController

quality, result, invasion = DetectController.analyse_scratch(
	exp_image,
	con_image=None,
)
```

输入图像要求为非空 RGB 数组，形状为 `(height, width, 3)`。

单帧结果 `ScratchResult` 包含：

- `area.pixel`：划痕掩膜非零像素数量
- `area.um`：物理面积，计算公式为 `area.pixel * um_per_pixel²`
- `width_avg`：有效图像行的平均划痕宽度，单位为像素
- `width_std`：划痕宽度标准差
- `roughness_left`、`roughness_right`：左右边界相对平滑边界的波动标准差
- `confluence`：细胞覆盖率，计算公式为 `1 - 划痕面积 / 图像总面积`

结果对象支持中文字符串输出，例如直接执行 `print(result)` 即可查看格式化结果。

### 边界提取

`ContouringGuassian` 对掩膜逐行提取最左和最右前景像素，随后对缺失行进行线性插值，并对边界进行高斯平滑，得到左右边界和中心线。

### 时序分析

入口：

```python
error_code, result_list, invasion_list = (
	DetectController.analyse_scratch_kinetic(
		exp_image_list=images,
		timestamp_list=timestamps,
		con_image_list=None,
	)
)
```

`result_list` 是与输入帧一一对应的 `list[ScratchResultKinetic]`。每一帧结果包含该帧原始单帧结果、愈合率、迁移速度和质量状态。

- 首帧 `heal_raw`、`heal_corrected` 和 `speed` 均为 `0`
- 原始愈合率：`(A0 - At) / A0`
- 迁移速度：`(W0 - Wt) / (2 * t)`，结果根据 `um_per_pixel` 换算为 μm/h
- `timestamp_list` 当前按秒处理，计算速度时转换为小时
- 提供对照组时，校正愈合率为实验组愈合率减去对照组愈合率

时序输入必须满足：实验组图像数量与时间戳数量一致；如果提供对照组，对照组数量也必须一致。输入异常时返回错误码 `1`，正常完成时返回 `0`。

### 侵入数据

提供对照图像时，控制器会生成与实验帧对应的 `invasion_list`。每项包含：

- `area`：参考划痕面积减去当前实验划痕面积后的非负面积
- `ratio`：侵入面积相对于参考面积的比例

未提供对照图像时，`invasion_list` 为 `None`。

## 量化指标与公式

设初始帧划痕面积为 $A_0$，当前帧划痕面积为 $A_t$，初始平均宽度为 $W_0$，当前平均宽度为 $W_t$，经过时间为 $t$：

| 指标 | 公式 | 当前单位或说明 |
| --- | --- | --- |
| 像素面积 | `countNonZero(mask)` | 像素数量 |
| 物理面积 | `A_px * um_per_pixel²` | μm² |
| 细胞覆盖率 | `1 - A_px / image_size` | 0 到 1 |
| 原始愈合率 | `(A0 - At) / A0` | 结果字段以比例保存 |
| 平均宽度 | `mean(rx - lx)` | 像素 |
| 宽度标准差 | `std(rx - lx)` | 像素 |
| 迁移速度 | `(W0 - Wt) * um_per_pixel / (2 * t)` | μm/h，`t` 为小时 |
| 侵入比例 | `invaded_area / reference_area` | 0 到 1 |

## 运行测试

在项目根目录执行：

```bash
pytest
```

也可以只运行控制器测试：

```bash
pytest test/test_scratch.py -q
```

测试使用 `data/input/s1` 中的文件名时间信息生成时间戳。文件名格式为：

```text
xxx_yyyymmdd_hhmmss_cellScratch.jpg
```

时间戳表示相对首帧的秒数，例如首帧为 `0`，之后的帧为相对于首帧的采集秒数。

## 已知限制与未完成的功能

- 骨架法轮廓提取 `ContouringSkeleton.process` 仍未实现。
- 增殖校正目前是简单的实验组愈合率减对照组愈合率，尚未实现设计文档中的完整增殖倍数模型、异常对照检测和质量标志位。
- 尚未实现可配置时间区间的线性回归迁移速率拟合。
- 当前结果只保存在内存对象中，尚未生成设计文档中提到的 CSV 动力学报告。
- `ScratchParameter.fill_hole` 尚未用于动态切换掩膜孔洞填充策略。
- 质量判断目前主要依据最终划痕面积是否为零，`ScratchQuality.SMALL` 尚未建立面积阈值和完整的置信度判断。
- 测试仍以样例和基础断言为主，边界输入、异常输入、对照组和数值公式的自动化覆盖不足。

## 代码使用边界

本项目当前是算法原型，不负责：

- 显微镜图像采集
- 帧间配准
- 图像质量增强或颜色校正
- 实验分组管理
- 结果数据库存储和可视化界面

这些工作应在调用控制器前或上层应用中完成。