from .result import ScratchQuality, ScratchArea, ScratchParameter, ScratchInvasionData, ScratchResult, ScratchResultKinetic
from .mask import MaskingNonEnvelop, MaskingEnvelop
from .contour import ContouringGuassian, ContouringSkeleton
import numpy as np

class DetectController:
    """划痕实验分析控制器

    提供单张图像划痕分析与时序序列划痕分析接口，
    完成划痕分割、前沿提取、量化指标计算、侵入数据输出。
    """

    @staticmethod
    def _result_from_mask(mask: np.ndarray, parameter: ScratchParameter) -> ScratchResult:
        mask = np.asarray(mask) > 0
        result = ScratchResult(area=ScratchArea())
        result.area.pixel = float(mask.sum())
        result.area.um = result.area.pixel * parameter.um_per_pixel ** 2
        result.confluence = float(1.0 - result.area.pixel / mask.size) if mask.size else 0.0

        left, right, _center, _debug = ContouringGuassian.process(mask.astype(np.uint8))
        valid = np.isfinite(left) & np.isfinite(right)
        if not np.any(valid):
            return result

        widths = right[valid] - left[valid]
        result.width_avg = float(np.mean(widths))
        result.width_std = float(np.std(widths))

        smooth_left, smooth_right = ContouringGuassian.smooth_boundaries(left, right)
        result.roughness_left = float(np.std(left[valid] - smooth_left[valid]))
        result.roughness_right = float(np.std(right[valid] - smooth_right[valid]))
        return result

    @staticmethod
    def _analyse_frame(exp_image: np.ndarray, parameter: ScratchParameter) -> ScratchResult:
        if exp_image is None:
            raise ValueError("exp_image cannot be None")
        image = np.asarray(exp_image)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("exp_image must be a non-empty RGB image")
        return DetectController._result_from_mask(
            MaskingNonEnvelop.process(image), parameter
        )

    @staticmethod
    def _healing_rate(start: ScratchResult, end: ScratchResult) -> float:
        if start.area is None or start.area.pixel <= 0:
            return 0.0
        return float((start.area.pixel - end.area.pixel) / start.area.pixel)

    @staticmethod
    def _invasion_data(result: ScratchResult, reference: ScratchResult = None) -> ScratchInvasionData:
        if reference is None or reference.area is None or reference.area.pixel <= 0:
            return ScratchInvasionData()
        invaded = max(0.0, reference.area.pixel - result.area.pixel)
        return ScratchInvasionData(
            area=ScratchArea(pixel=invaded, um=invaded * (result.area.um / result.area.pixel) if result.area.pixel else 0.0),
            ratio=float(invaded / reference.area.pixel),
        )

    @staticmethod
    def analyse_scratch(
        exp_image: np.ndarray,
        con_image: np.ndarray = None,
        parameter: ScratchParameter = None
    ) -> tuple[ScratchQuality, ScratchResult, ScratchInvasionData]:
        """划痕实验单张图像分析接口

        对输入明场/相差图像执行划痕分割、前沿提取与量化指标计算。

        Parameters
        ----------
        exp_image : np.ndarray
            输入实验组单张图像，不可为空
        con_image : np.ndarray
            输入对照组单张图像，可为空
        parameter : ScratchParameter
            分析输入参数结构体

        Returns
        -------
        tuple[ScratchQuality, ScratchResult, ScratchInvasionData]
            (quality, result, invasion_data)

            quality : ScratchQuality
                划痕质量枚举
            result : ScratchResult
                划痕分析输出结果结构体
            invasion_data : ScratchInvasionData
                划痕区细胞侵入结果结构体

        Notes
        -----
        - 输入图像需预处理完成；
        - 内部不做帧间配准，时序序列需上层完成配准后逐帧调用。
        """
        parameter = parameter or ScratchParameter()
        result = ScratchResult(area=ScratchArea())

        exp_image_mask = MaskingNonEnvelop.process(exp_image)

        # 通过image_mask统计有多少个像素点
        result.area.pixel = float(np.count_nonzero(exp_image_mask))
        result.area.um = result.area.pixel * parameter.um_per_pixel ** 2
        # 通过对比mask和image总的像素点，计算出细胞覆盖率
        result.confluence = float(
            1.0 - (result.area.pixel / exp_image_mask.size)
        ) if exp_image_mask.size else 0.0

        exp_image_left, exp_image_right, exp_image_center, _ = ContouringGuassian.process(
            mask=exp_image_mask
        )

        # 通过 exp_image_left, exp_image_right 计算宽度平均值和方差，以及左右两边的粗糙度
        valid = np.isfinite(exp_image_left) & np.isfinite(exp_image_right)
        if np.any(valid):
            widths = exp_image_right[valid] - exp_image_left[valid]
            result.width_avg = float(np.mean(widths))
            result.width_std = float(np.std(widths))
            smooth_left, smooth_right = ContouringGuassian.smooth_boundaries(
                exp_image_left, exp_image_right
            )
            result.roughness_left = float(
                np.std(exp_image_left[valid] - smooth_left[valid])
            )
            result.roughness_right = float(
                np.std(exp_image_right[valid] - smooth_right[valid])
            )

        invasion = None
        if con_image is not None:
            control = DetectController._analyse_frame(con_image, parameter)
            invasion = DetectController._invasion_data(result, control)
        quality = ScratchQuality.ABNORMAL if result.area.pixel == 0 else ScratchQuality.NORMAL
        return quality, result, invasion


    def analyse_scratch_kinetic(
        exp_image_list: np.ndarray,
        con_image_list: np.ndarray = None,
        timestamp_list: np.ndarray = None,
        parameter: ScratchParameter = None
    ) -> tuple[int, list[ScratchResultKinetic], list[ScratchInvasionData] | None]:
        """划痕实验时序图像分析接口

        对输入明场/相差时序图像执行划痕分割、前沿提取与量化指标计算。

        Parameters
        ----------
        exp_image_list : np.ndarray
            输入实验组图像数组，不可为空；数组每个元素为单张图像
        con_image_list : np.ndarray
            输入对照组图像数组，可为空；数组每个元素为单张图像
        timestamp_list : np.ndarray
            输入图像对应时间戳数组，不可为空
        parameter : ScratchParameter
            分析输入参数结构体，不可为空

        Returns
        -------
        tuple[int, list[ScratchResultKinetic], list[ScratchInvasionData] | None]
            (error_code, result_kinetic, invasion_data)

            error_code : int
                错误码；0代表处理成功；非0代表异常（图像为空、分割失败等）
            result_kinetic : list[ScratchResultKinetic]
                时序划痕分析输出结果结构体
            invasion_data : list[ScratchInvasionData]
                时序划痕区细胞侵入结果结构体

        Notes
        -----
        - 如果传入有效 con_image_list，则执行增值校正计算；
        - 输入图像数量由输入数组维度决定，实验组、对照组、时间戳数组长度需要对齐。
        """
        parameter = parameter or ScratchParameter()
        if exp_image_list is None or timestamp_list is None:
            return 1, [], None

        try:
            frames = list(exp_image_list)
            timestamps = np.asarray(timestamp_list, dtype=float).reshape(-1)
            if not frames or len(frames) != len(timestamps):
                return 1, [], None
            raw_results = [DetectController._analyse_frame(frame, parameter) for frame in frames]

            initial = raw_results[0]
            initial_area = initial.area.pixel if initial.area is not None else 0.0
            initial_width = initial.width_avg
            kinetic_results = []
            for index, result in enumerate(raw_results):
                kinetic = ScratchResultKinetic(raw=result)
                elapsed_seconds = float(timestamps[index] - timestamps[0])
                if index > 0 and initial_area > 0:
                    kinetic.heal_raw = float(
                        (initial_area - result.area.pixel) / initial_area
                    )
                if index > 0 and elapsed_seconds > 0:
                    elapsed_hours = elapsed_seconds / 3600.0
                    kinetic.speed = float(
                        (initial_width - result.width_avg)
                        * parameter.um_per_pixel
                        / (2.0 * elapsed_hours)
                    )
                kinetic.quality = (
                    ScratchQuality.ABNORMAL
                    if result.area is None or result.area.pixel == 0
                    else ScratchQuality.NORMAL
                )
                kinetic_results.append(kinetic)

            invasion_results = None
            if con_image_list is not None:
                control_frames = list(con_image_list)
                if len(control_frames) != len(frames):
                    return 1, [], None
                control_results = [DetectController._analyse_frame(frame, parameter) for frame in control_frames]
                control_initial = control_results[0]
                control_area = (
                    control_initial.area.pixel
                    if control_initial.area is not None
                    else 0.0
                )
                invasion_results = [
                    DetectController._invasion_data(result, control_result)
                    for result, control_result in zip(raw_results, control_results)
                ]
                for index, (kinetic, control_result) in enumerate(
                    zip(kinetic_results, control_results)
                ):
                    if index > 0 and control_area > 0:
                        control_heal = (
                            control_area - control_result.area.pixel
                        ) / control_area
                        kinetic.heal_corrected = kinetic.heal_raw - control_heal
            else:
                for kinetic in kinetic_results:
                    kinetic.heal_corrected = kinetic.heal_raw
            return 0, kinetic_results, invasion_results
        except (TypeError, ValueError, IndexError):
            return 1, [], None
