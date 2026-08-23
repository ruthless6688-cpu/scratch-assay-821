from dataclasses import dataclass
from enum import IntEnum

class ScratchQuality(IntEnum):
    NORMAL = 0
    SMALL = 1
    ABNORMAL = 2

@dataclass
class ScratchParameter:
    fill_hole: bool = True
    um_per_pixel: float = 1.0

@dataclass
class ScratchArea:
    pixel: float = 0.0
    um: float = 0.0

    def __str__(self) -> str:
        return f"像素面积: {self.pixel:.2f}\n物理面积: {self.um:.2f} 平方微米"

@dataclass
class ScratchInvasionData:
    area: ScratchArea | None = None
    ratio: float = 0.0

    def __str__(self) -> str:
        area = str(self.area) if self.area is not None else "无"
        return f"侵入面积:\n{area}\n侵入比例: {self.ratio:.2%}"     

@dataclass
class ScratchResult:
    area: ScratchArea | None = None
    width_avg: float = 0.0
    width_std: float = 0.0
    roughness_left: float = 0.0
    roughness_right: float = 0.0
    confluence: float = 0.0

    def __str__(self) -> str:
        area = str(self.area) if self.area is not None else "无"
        return (
            f"{area}\n"
            f"平均宽度: {self.width_avg:.2f} 像素\n"
            f"宽度标准差: {self.width_std:.2f} \n"
            f"左边界粗糙度: {self.roughness_left:.2f} %\n"
            f"右边界粗糙度: {self.roughness_right:.2f} %\n"
            f"细胞覆盖率: {self.confluence:.2%}"
        )


@dataclass
class ScratchResultKinetic:
    raw: ScratchResult | None = None
    heal_raw: float = 0.0
    heal_corrected: float = 0.0
    speed: float = 0.0
    quality: ScratchQuality = ScratchQuality.NORMAL

    def __str__(self) -> str:
        quality_names = {
            ScratchQuality.NORMAL: "正常",
            ScratchQuality.SMALL: "划痕过小",
            ScratchQuality.ABNORMAL: "异常",
        }
        raw = str(self.raw) if self.raw is not None else "无"
        quality = quality_names.get(self.quality, str(self.quality))
        return (
            f"{raw}\n"
            f"原始愈合率: {self.heal_raw:.2%}\n"
            f"校正后愈合率: {self.heal_corrected:.2%}\n"
            f"愈合速度: {self.speed:.6f}\n"
            f"质量: {quality}"
        )

