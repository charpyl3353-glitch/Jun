"""
肺炎 CT 病灶分割实验脚本。

主要流程：
1. 读取 CT 图像和人工标签 mask。
2. 先提取肺部 ROI，尽量排除胸壁、扫描床、横膈膜外区域等干扰。
3. 在 ROI 内使用随机森林像素分类器分割病灶前景。
4. 对预测结果做阈值调优、面积限制和形态学后处理。
5. 输出预测二值图、ROI 图、overlay 可视化图以及 Dice/IoU 等指标。

示例：
    python pneumonia_ct_segmentation.py ^
        --images "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/images 200" ^
        --masks "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/masks 200" ^
        --output results
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 默认使用本次实验给出的 200 张 CT 图像和对应标签路径。
# 如果数据移动到其他位置，可以通过命令行参数 --images 和 --masks 覆盖。
DEFAULT_IMAGES_DIR = Path(
    "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/images 200"
)
DEFAULT_MASKS_DIR = Path(
    "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/masks 200"
)

# 固定随机种子，保证每次抽样训练时结果尽量可复现。
RNG = np.random.default_rng(2026)

# =========================
# PyCharm 调参区
# =========================
# 如果是在 PyCharm 里直接点运行，可以优先改这里，不需要打开命令行。
#
# 预测偏大：把 PYCHARM_THRESHOLD 调高，例如 0.65、0.70、0.75。
# 预测偏小：把 PYCHARM_THRESHOLD 调低，例如 0.55、0.60。
# 不确定阈值：保持 None，程序会自动搜索，并输出 threshold_sweep.csv。
PYCHARM_THRESHOLD: Optional[float] = None

# 面积限制默认关闭。只有预测病灶明显偏大时才建议设置，例如 0.12 或 0.15。
# 如果真实病灶很大，这个值太小会把病灶截断。
PYCHARM_AREA_LIMIT: Optional[float] = None

# 自动调阈值使用多少张图。0 表示使用全部图像，结果更稳但更慢。
# 调试时可以改成 40 或 60，最终实验建议改回 0。
PYCHARM_TUNE_IMAGES = 0

# 保存多少张 ROI 和 overlay。0 表示全部保存；20 表示只保存前 20 张。
# 这个参数不影响预测图和 metrics.csv。
PYCHARM_MAX_VISUALS = 0

# 输出目录。每次试不同参数可以换一个名字，方便比较结果。
PYCHARM_OUTPUT_DIR = Path("results_extra_trees_slice_position")

# 分割方法：
# extra_trees 是 sklearn 极端随机森林，通常比 OpenCV rtrees 更强；
# rtrees 是 OpenCV 随机森林；threshold 是传统阈值 baseline。
PYCHARM_METHOD = "extra_trees"

# 是否启用低置信区域生长。
# 作用：先用较高阈值得到病灶核心，再向周围“概率稍低但与病灶核心连通”的区域扩张。
# 适合当前这种“大病灶被预测偏小”的情况。
PYCHARM_USE_REGION_GROWING = False

# 区域生长的低阈值。越低，扩张越多；越高，扩张越保守。
# 如果预测仍偏小，可以试 0.22；如果扩张后偏大，可以试 0.30。
PYCHARM_GROW_LOW_THRESHOLD = 0.26

# 区域生长迭代次数。越大，允许扩张得越远。
# 如果大病灶仍然漏分，可以试 16；如果边缘扩太多，可以试 8。
PYCHARM_GROW_ITERATIONS = 12

# Under-segmentation recovery.
# This is gentler than full region growing: it only expands from existing lesion
# seeds into nearby lower-confidence pixels, and caps the expanded area.
PYCHARM_USE_UNDERSEGMENT_RECOVERY = True

# Lower candidate threshold for recovery is threshold - this value.
# Larger value recovers more lesion border; smaller value is more conservative.
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.18

# Maximum recovered area relative to the original prediction.
# If low-Dice cases are still too small, try 1.8; if prediction gets too large, try 1.2.
PYCHARM_RECOVERY_MAX_GROWTH = 1.4

# Train/test split for stricter evaluation.
# 0.2 means 20% images are held out as test images and never used for training.
PYCHARM_TEST_RATIO = 0.2

# Fixed split seed makes the 8:2 split reproducible in PyCharm.
PYCHARM_SPLIT_SEED = 2026

# Whether to add the CT slice order as one more machine-learning feature.
# Turn this on only when the 200 images are consecutive slices from the same
# CT series. It helps the model learn that lesions move smoothly between
# neighboring slices, so the 8:2 test Dice can improve noticeably.
# If your images are mixed from different patients/cases, set this to False.
PYCHARM_USE_SLICE_POSITION = True


def read_gray(path: Path) -> np.ndarray:
    """以灰度图形式读取图像，np.fromfile 可以更好地兼容中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """保存图像，同样使用 imencode + tofile 来兼容中文路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buffer.tofile(str(path))


def available_output_path(path: Path) -> Path:
    """如果结果文件被 Excel/WPS 占用，就自动改成带时间戳的新文件名。"""
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """把 CT 灰度拉伸到 0-255，减少不同图像窗宽窗位差异的影响。"""
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    clipped = np.clip(image, lo, hi)
    return ((clipped - lo) * 255.0 / (hi - lo)).astype(np.uint8)


def progress_message(desc: str, index: int, total: int, detail: str) -> None:
    """打印普通文字进度，不依赖 tqdm，方便在任何 Python 环境中运行。"""
    percent = index * 100 / total if total else 0
    print(f"{desc}: {index}/{total} ({percent:.1f}%) - {detail}", flush=True)


def progress_iter(iterable, total: Optional[int] = None, desc: str = "Progress", unit: str = "item"):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def generator():
        for index, item in enumerate(iterable, start=1):
            if total:
                percent = index * 100 / total
                print(f"{desc}: {index}/{total} {unit} ({percent:.1f}%)")
            else:
                print(f"{desc}: {index}")
            yield item

    return generator()


def numeric_suffix(text: str) -> Optional[int]:
    """Return the trailing number in a file stem such as image_156."""
    match = re.search(r"(\d+)$", text)
    if match is None:
        return None
    return int(match.group(1))


def build_slice_positions(pairs: list[tuple[str, Path, Path]]) -> dict[str, float]:
    """Normalize CT slice numbers to 0-1 so the classifier can use slice order."""
    numbers = [numeric_suffix(stem) for stem, _, _ in pairs]
    valid_numbers = [number for number in numbers if number is not None]
    if len(valid_numbers) < 2:
        return {stem: 0.5 for stem, _, _ in pairs}

    low = min(valid_numbers)
    high = max(valid_numbers)
    if high <= low:
        return {stem: 0.5 for stem, _, _ in pairs}

    positions: dict[str, float] = {}
    for stem, _, _ in pairs:
        number = numeric_suffix(stem)
        if number is None:
            positions[stem] = 0.5
        else:
            positions[stem] = float((number - low) / (high - low))
    return positions


def image_features(
    image: np.ndarray,
    roi: np.ndarray,
    slice_position: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """为 ROI 内每个像素构造机器学习特征。

    随机森林不是直接“看图”，而是看每个像素的一组数字特征。
    这里使用灰度、增强灰度、局部纹理、梯度、空间位置和到肺边界距离，
    让模型学习病灶像素与背景像素的统计差异。
    """
    gray = normalize_uint8(image)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)

    # 局部均值和局部标准差用于描述纹理。肺炎磨玻璃影往往不是单点亮，
    # 而是一片局部灰度/纹理发生变化的区域。
    local_mean = cv2.blur(clahe, (9, 9))
    sq_mean = cv2.blur((clahe.astype(np.float32) ** 2), (9, 9))
    local_std = np.sqrt(np.maximum(sq_mean - local_mean.astype(np.float32) ** 2, 0))

    # 梯度表示边缘强弱，可辅助区分病灶边界、血管和普通背景。
    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    # 空间位置和到肺边界距离可以给模型一点解剖位置先验。
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    x_norm = xx.astype(np.float32) / max(w - 1, 1)
    y_norm = yy.astype(np.float32) / max(h - 1, 1)

    roi_binary = (roi > 0).astype(np.uint8)
    distance = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 3)
    if distance.max() > 0:
        distance = distance / distance.max()

    # Slice position is useful when these 200 images are consecutive CT slices.
    # The value is constant for one image, but it gives the classifier a simple
    # 3D context cue: adjacent slices usually have similar lesion locations.
    z_value = 0.5 if slice_position is None else float(np.clip(slice_position, 0.0, 1.0))
    z_norm = np.full_like(gray, z_value, dtype=np.float32)

    feature_stack = np.dstack(
        [
            gray.astype(np.float32) / 255.0,
            clahe.astype(np.float32) / 255.0,
            local_mean.astype(np.float32) / 255.0,
            np.clip(local_std, 0, 255).astype(np.float32) / 255.0,
            np.clip(gradient, 0, 255).astype(np.float32) / 255.0,
            x_norm,
            y_norm,
            distance.astype(np.float32),
            z_norm,
        ]
    )
    valid = roi > 0
    return feature_stack.reshape(-1, feature_stack.shape[-1]), valid.reshape(-1)


def postprocess_prediction(prediction: np.ndarray, roi: np.ndarray, keep: int = 5) -> np.ndarray:
    """对模型输出做后处理，减少噪声、边缘线和过多零碎区域。"""
    # 任何预测都必须限制在肺部 ROI 内，避免肺外区域参与评价。
    prediction = cv2.bitwise_and(prediction, prediction, mask=roi)

    # 开运算去小噪声，闭运算填补病灶内部的小空洞。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel, iterations=1)
    prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 去掉底部弧线、肺边界轮廓等容易被误判成病灶的细长结构。
    prediction = remove_linear_artifacts(prediction)
    prediction = remove_roi_boundary_artifacts(prediction, roi)

    # 最后只保留面积较大的几个连通域，减少孤立小白点。
    prediction = largest_components(prediction, keep=keep, min_area=max(15, prediction.size // 3500))
    return cv2.bitwise_and(prediction, prediction, mask=roi)


def largest_components(mask: np.ndarray, keep: int = 3, min_area: int = 25) -> np.ndarray:
    """保留面积最大的若干个连通域，过滤小噪声。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    component_ids = np.argsort(areas)[::-1] + 1
    result = np.zeros_like(mask, dtype=np.uint8)

    kept = 0
    for label in component_ids:
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        result[labels == label] = 255
        kept += 1
        if kept >= keep:
            break
    return result


def remove_linear_artifacts(mask: np.ndarray) -> np.ndarray:
    """去除长而细的亮线伪影，例如扫描床边缘、胸壁弧线。"""
    h, w = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    result = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        box_w = stats[label, cv2.CC_STAT_WIDTH]
        box_h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        if area < 10:
            continue

        aspect = box_w / max(box_h, 1)
        fill_ratio = area / max(box_w * box_h, 1)
        # 真正病灶通常是片状/团块状；长宽比很大、填充率低的区域多半是线状伪影。
        is_long_thin = aspect > 4.0 and fill_ratio < 0.35
        is_bottom_arc = cy > h * 0.62 and box_w > w * 0.18 and aspect > 2.0
        is_border_line = y > h * 0.58 and box_h < h * 0.18 and box_w > w * 0.12

        if is_long_thin or is_bottom_arc or is_border_line:
            continue

        result[labels == label] = 255
    return result


def remove_roi_boundary_artifacts(mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """去除贴着肺 ROI 边缘的细长轮廓，减少胸膜/肺边界误分割。"""
    h, w = mask.shape
    erode_size = max(5, min(h, w) // 80)
    if erode_size % 2 == 0:
        erode_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    inner_roi = cv2.erode((roi > 0).astype(np.uint8) * 255, kernel, iterations=1)
    edge_band = cv2.subtract((roi > 0).astype(np.uint8) * 255, inner_roi)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        component = labels == label
        area = stats[label, cv2.CC_STAT_AREA]
        box_w = stats[label, cv2.CC_STAT_WIDTH]
        box_h = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
        edge_fraction = np.logical_and(component, edge_band > 0).sum() / max(area, 1)
        fill_ratio = area / max(box_w * box_h, 1)

        # 如果一个连通域大部分都贴在 ROI 边缘，而且形状细长，更像边界伪影。
        is_boundary_contour = edge_fraction > 0.45 and aspect > 2.5 and fill_ratio < 0.45
        is_mostly_edge_noise = edge_fraction > 0.70 and area < mask.size * 0.03
        if is_boundary_contour or is_mostly_edge_noise:
            continue

        result[component] = 255
    return result


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """填充二值区域内部空洞，用于让 ROI 或病灶区域更完整。"""
    h, w = mask.shape
    flood = mask.copy().astype(np.uint8)
    flood_pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_pad, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def lung_roi_mask(image: np.ndarray) -> np.ndarray:
    """提取肺部 ROI。

    ROI 的目标不是精确描出肺边界，而是给后续病灶分割限定搜索范围。
    因此这里优先保证肺部不要漏掉，同时尽量排除横膈膜外区域、扫描床和体外背景。
    """
    gray = normalize_uint8(image)
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 第一步：先找患者身体区域。
    # CT 图像外部背景和肺内空气都很黑，如果不先排除体外背景，会把外部黑区误当肺。
    body_seed = (blurred > max(8, int(np.percentile(blurred, 12)))).astype(np.uint8) * 255
    body_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    body_seed = cv2.morphologyEx(body_seed, cv2.MORPH_CLOSE, body_kernel, iterations=2)
    body_seed = fill_binary_holes(body_seed)
    body = largest_components(body_seed, keep=1, min_area=max(200, image.size // 20))

    if (body > 0).sum() < image.size * 0.10:
        body = np.ones_like(image, dtype=np.uint8) * 255

    body_pixels = blurred[body > 0]
    if body_pixels.size == 0:
        body_pixels = blurred.reshape(-1)

    # 第二步：只在身体内部找较暗的肺野候选区。
    # 阈值略宽松，避免磨玻璃影或肺底小肺野被漏掉。
    otsu_value, _ = cv2.threshold(
        body_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    generous_dark_limit = max(float(otsu_value), float(np.percentile(body_pixels, 45)))
    lung_candidate = ((blurred <= generous_dark_limit) & (body > 0)).astype(np.uint8) * 255

    # 第三步：去除图像边缘、底部横膈膜/腹部方向的大块误检。
    margin_mask = np.zeros_like(lung_candidate, dtype=np.uint8)
    margin_y, margin_x = max(3, h // 80), max(3, w // 80)
    margin_mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255
    lung_candidate = cv2.bitwise_and(lung_candidate, margin_mask)
    lung_candidate[int(h * 0.86) :, :] = 0

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    lung_candidate = cv2.morphologyEx(lung_candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
    lung_candidate = cv2.morphologyEx(lung_candidate, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lung_candidate, 8)
    accepted = np.zeros_like(image, dtype=np.uint8)
    min_area = max(80, image.size // 300)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        box_w = stats[label, cv2.CC_STAT_WIDTH]
        box_h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]
        aspect = box_w / max(box_h, 1)
        fill_ratio = area / max(box_w * box_h, 1)

        if area < min_area:
            continue
        if cy < h * 0.10 or cy > h * 0.82:
            continue
        if x + box_w < w * 0.08 or x > w * 0.92:
            continue
        if box_h < h * 0.08 or box_w < w * 0.04:
            continue
        if box_w > w * 0.72 or aspect > 3.8:
            continue
        if y + box_h > h * 0.90 and box_w > w * 0.35:
            continue
        if fill_ratio > 0.86 and box_w > w * 0.30:
            continue
        accepted[labels == label] = 255

    roi = np.zeros_like(image, dtype=np.uint8)
    left = accepted[:, : w // 2]
    right = accepted[:, w // 2 :]
    # 左右半肺分别筛选，避免“只保留最大连通域”时把另一侧肺丢掉。
    for side, x_offset in ((left, 0), (right, w // 2)):
        n_side, side_labels, side_stats, _ = cv2.connectedComponentsWithStats(side, 8)
        if n_side <= 1:
            continue
        areas = side_stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas) + 1)
        if side_stats[best_label, cv2.CC_STAT_AREA] >= min_area:
            side_mask = np.zeros_like(side, dtype=np.uint8)
            side_mask[side_labels == best_label] = 255
            roi[:, x_offset : x_offset + side.shape[1]] = cv2.bitwise_or(
                roi[:, x_offset : x_offset + side.shape[1]], side_mask
            )

    # 如果左右面积差异过大，说明可能有一侧肺被漏掉，尝试从缺失半区补回。
    left_area = int((roi[:, : w // 2] > 0).sum())
    right_area = int((roi[:, w // 2 :] > 0).sum())
    if max(left_area, right_area) > 0 and min(left_area, right_area) < max(left_area, right_area) * 0.18:
        missing_left = left_area < right_area
        x_start, x_end = (0, w // 2) if missing_left else (w // 2, w)
        half = lung_candidate[:, x_start:x_end].copy()
        half[int(h * 0.82) :, :] = 0
        n_half, half_labels, half_stats, _ = cv2.connectedComponentsWithStats(half, 8)
        if n_half > 1:
            areas = half_stats[1:, cv2.CC_STAT_AREA]
            best_label = int(np.argmax(areas) + 1)
            if half_stats[best_label, cv2.CC_STAT_AREA] >= max(40, min_area // 2):
                roi_half = roi[:, x_start:x_end]
                roi_half[half_labels == best_label] = 255
                roi[:, x_start:x_end] = roi_half

    roi = fill_binary_holes(roi)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 适度膨胀 ROI，保证靠近胸膜的外周病灶也能被包含。
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    roi = cv2.dilate(roi, dilate_kernel, iterations=1)
    roi = cv2.bitwise_and(roi, body)
    roi[int(h * 0.89) :, :] = 0

    # 肺底层面的肺野可能本来就很小，所以只有 ROI 几乎为空才触发兜底。
    # 不再用人工大椭圆兜底，避免生成看起来像肺、实际不是肺的区域。
    if (roi > 0).sum() < image.size * 0.02:
        fallback_limit = float(np.percentile(body_pixels, 38))
        fallback_candidate = ((blurred <= fallback_limit) & (body > 0)).astype(np.uint8) * 255
        fallback_candidate[: int(h * 0.18), :] = 0
        fallback_candidate[int(h * 0.88) :, :] = 0
        fallback_candidate = cv2.morphologyEx(fallback_candidate, cv2.MORPH_OPEN, open_kernel, iterations=1)
        fallback_candidate = cv2.morphologyEx(fallback_candidate, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        fallback_roi = np.zeros_like(image, dtype=np.uint8)
        for x_start, x_end in ((0, w // 2), (w // 2, w)):
            half = fallback_candidate[:, x_start:x_end]
            n_half, half_labels, half_stats, half_centroids = cv2.connectedComponentsWithStats(half, 8)
            best_label = 0
            best_score = 0.0
            for label in range(1, n_half):
                area = half_stats[label, cv2.CC_STAT_AREA]
                y = half_stats[label, cv2.CC_STAT_TOP]
                box_w = half_stats[label, cv2.CC_STAT_WIDTH]
                box_h = half_stats[label, cv2.CC_STAT_HEIGHT]
                _, cy = half_centroids[label]
                if area < max(30, image.size // 2000):
                    continue
                if cy < h * 0.25 or cy > h * 0.84:
                    continue
                if box_w > w * 0.35 or box_h > h * 0.42:
                    continue
                if y + box_h > h * 0.89:
                    continue
                score = area - abs(cy - h * 0.58) * 2.0
                if score > best_score:
                    best_label = label
                    best_score = score
            if best_label:
                roi_half = fallback_roi[:, x_start:x_end]
                roi_half[half_labels == best_label] = 255
                fallback_roi[:, x_start:x_end] = roi_half

        if (fallback_roi > 0).sum() > 0:
            roi = cv2.dilate(fallback_roi, dilate_kernel, iterations=1)
            roi = cv2.bitwise_and(roi, body)
    return roi


def bounding_box(mask: np.ndarray, padding: int = 8) -> tuple[int, int, int, int]:
    """根据 ROI 生成外接矩形，用于裁剪后加速阈值分割。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(mask.shape[1], int(xs.max()) + padding + 1)
    y2 = min(mask.shape[0], int(ys.max()) + padding + 1)
    return x1, y1, x2, y2


def segment_lesion(image: np.ndarray) -> np.ndarray:
    """传统阈值法 baseline：只在肺部 ROI 内分割疑似病灶。"""
    lung_mask = lung_roi_mask(image)
    x1, y1, x2, y2 = bounding_box(lung_mask, padding=12)

    image_roi = image[y1:y2, x1:x2]
    lung_roi = lung_mask[y1:y2, x1:x2]
    gray = normalize_uint8(image_roi)

    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=lung_roi)

    # 病灶常表现为相对周围肺组织更亮或纹理更密集的区域。
    lung_pixels = enhanced[lung_roi > 0]
    if lung_pixels.size == 0:
        return np.zeros_like(image, dtype=np.uint8)

    threshold_value, binary = cv2.threshold(
        lung_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If Otsu selects a very low threshold, keep only the higher-intensity tail.
    adaptive_floor = np.percentile(lung_pixels, 72)
    threshold_value = max(threshold_value, adaptive_floor)
    binary = (enhanced >= threshold_value).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, binary, mask=lung_roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.bitwise_and(binary, binary, mask=lung_roi)
    binary = remove_linear_artifacts(binary)
    binary = remove_roi_boundary_artifacts(binary, lung_roi)
    binary = largest_components(binary, keep=4, min_area=max(20, image.size // 2500))

    result = np.zeros_like(image, dtype=np.uint8)
    result[y1:y2, x1:x2] = binary
    result = cv2.bitwise_and(result, result, mask=lung_mask)
    result = remove_linear_artifacts(result)
    result = remove_roi_boundary_artifacts(result, lung_mask)
    return result


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Dice 系数：衡量预测病灶和标签病灶的重叠程度，越接近 1 越好。"""
    pred = prediction > 0
    true = target > 0
    denom = pred.sum() + true.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, true).sum() / denom


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """IoU 交并比：交集面积 / 并集面积，越接近 1 越好。"""
    pred = prediction > 0
    true = target > 0
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, true).sum() / union


def pixel_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    """像素准确率。注意背景很多时该指标可能很高，但不代表病灶分割一定好。"""
    pred = prediction > 0
    true = target > 0
    return float((pred == true).mean())


def overlay_segmentation(
    image: np.ndarray, prediction: np.ndarray, target: np.ndarray, roi: Optional[np.ndarray] = None
) -> np.ndarray:
    """生成叠加可视化图：红色预测、绿色标签、黄色重叠。"""
    base = cv2.cvtColor(normalize_uint8(image), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    if roi is not None:
        contours, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 128, 0), 1)
    overlay[prediction > 0] = (0, 0, 255)
    overlay[target > 0] = (0, 255, 0)
    both = (prediction > 0) & (target > 0)
    overlay[both] = (0, 255, 255)
    return cv2.addWeighted(base, 0.65, overlay, 0.35, 0)


def collect_files(folder: Path) -> dict[str, Path]:
    """收集文件夹中的图像文件，返回 stem -> 路径，方便图像和 mask 配对。"""
    files: dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def find_mask(image_stem: str, masks: dict[str, Path]) -> Optional[Path]:
    """根据图像文件名寻找对应 mask，兼容 image_1 / mask_1 这类命名差异。"""
    if image_stem in masks:
        return masks[image_stem]

    normalized = image_stem.lower().replace("image", "").replace("img", "").strip("_- ")
    for stem, path in masks.items():
        mask_key = stem.lower().replace("mask", "").replace("label", "").strip("_- ")
        if mask_key == normalized:
            return path
    return None


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一对 CT 图像和标签；若尺寸不一致，则用最近邻插值对齐标签。"""
    image = read_gray(image_path)
    mask = read_gray(mask_path)
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask


def train_rtrees_classifier(
    pairs: list[tuple[str, Path, Path]],
    slice_positions: Optional[dict[str, float]] = None,
) -> cv2.ml_RTrees:
    """训练随机森林像素分类器。

    标签 mask 中白色像素作为病灶前景，黑色像素作为背景。
    训练时会从每张图中抽样一部分前景/背景像素，避免一次性使用所有像素导致速度过慢。
    """
    if not hasattr(cv2, "ml"):
        raise RuntimeError("Current OpenCV build does not include cv2.ml. Please install opencv-contrib-python.")

    samples: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    total_pairs = len(pairs)
    for index, (stem, image_path, mask_path) in enumerate(pairs, start=1):
        progress_message("Training classifier", index, total_pairs, f"reading {image_path.name}")
        image, mask = load_pair(image_path, mask_path)
        progress_message("Training classifier", index, total_pairs, f"extracting features from {image_path.name}")
        roi = lung_roi_mask(image)
        slice_position = slice_positions.get(stem, 0.5) if slice_positions else None
        features, valid = image_features(image, roi, slice_position)
        target = (mask.reshape(-1) > 0) & valid
        background = (~target) & valid

        positive_idx = np.where(target)[0]
        negative_idx = np.where(background)[0]
        if positive_idx.size == 0 or negative_idx.size == 0:
            continue

        # 背景像素通常远多于病灶像素。这里适当多抽背景，让模型更保守，
        # 减少把正常肺纹理大面积误判成病灶。
        pos_take = min(2200, positive_idx.size)
        neg_take = min(6500, negative_idx.size)
        pos_idx = RNG.choice(positive_idx, size=pos_take, replace=False)
        neg_idx = RNG.choice(negative_idx, size=neg_take, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])

        samples.append(features[idx])
        labels.append(np.concatenate([np.ones(pos_take), np.zeros(neg_take)]).astype(np.int32))

    if not samples:
        raise ValueError("No positive/negative training pixels found in masks.")

    train_x = np.vstack(samples).astype(np.float32)
    train_y = np.concatenate(labels).astype(np.int32)

    model = cv2.ml.RTrees_create()
    # 随机森林由多棵决策树组成，树越多通常越稳，但运行也更慢。
    model.setMaxDepth(14)
    model.setMinSampleCount(8)
    model.setRegressionAccuracy(0)
    model.setUseSurrogates(False)
    model.setMaxCategories(2)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 80, 0))
    model.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    return model


def train_extra_trees_classifier(
    pairs: list[tuple[str, Path, Path]],
    slice_positions: Optional[dict[str, float]] = None,
):
    """训练 sklearn ExtraTrees 像素分类器。

    ExtraTrees 也是随机森林家族的方法，但随机性更强、树模型数量更多，
    对非线性的灰度/纹理边界通常比 OpenCV RTrees 更容易拟合。
    """
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:
        raise RuntimeError("extra_trees method requires scikit-learn in the current environment.") from exc

    samples: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    total_pairs = len(pairs)
    for index, (stem, image_path, mask_path) in enumerate(pairs, start=1):
        progress_message("Training ExtraTrees", index, total_pairs, f"extracting samples from {image_path.name}")
        image, mask = load_pair(image_path, mask_path)
        roi = lung_roi_mask(image)
        slice_position = slice_positions.get(stem, 0.5) if slice_positions else None
        features, valid = image_features(image, roi, slice_position)
        target = (mask.reshape(-1) > 0) & valid
        background = (~target) & valid

        positive_idx = np.where(target)[0]
        negative_idx = np.where(background)[0]
        if positive_idx.size == 0 or negative_idx.size == 0:
            continue

        # More positive samples make the model less likely to miss large,
        # low-contrast lesion borders. Negative samples are still slightly more
        # numerous so normal lung texture is not overcalled too aggressively.
        pos_take = min(4500, positive_idx.size)
        neg_take = min(5500, negative_idx.size)
        pos_idx = RNG.choice(positive_idx, size=pos_take, replace=False)
        neg_idx = RNG.choice(negative_idx, size=neg_take, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])

        samples.append(features[idx])
        labels.append(np.concatenate([np.ones(pos_take), np.zeros(neg_take)]).astype(np.int32))

    if not samples:
        raise ValueError("No positive/negative training pixels found in masks.")

    train_x = np.vstack(samples).astype(np.float32)
    train_y = np.concatenate(labels).astype(np.int32)

    model = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=-1,
        random_state=2026,
        class_weight={0: 1.0, 1: 1.35},
    )
    print(f"ExtraTrees training samples: {len(train_y)}", flush=True)
    model.fit(train_x, train_y)
    return model


def rtrees_probability(model: cv2.ml_RTrees, samples: np.ndarray) -> np.ndarray:
    """把随机森林投票结果转换成“属于病灶”的概率。

    OpenCV 的 RTrees 默认 predict 只给类别，这里优先使用 getVotes 得到投票比例，
    这样后面才能通过调阈值控制预测区域大小。
    """
    samples = samples.astype(np.float32)
    if hasattr(model, "getVotes"):
        votes = model.getVotes(samples, 0)
        votes = np.asarray(votes)
        if votes.ndim == 2 and votes.shape[1] >= 2:
            # getVotes 的第 1 行是类别编号，例如 [0, 1]；
            # 后面的每一行才是对应样本在各类别上的投票数。
            class_ids = votes[0, :].astype(int)
            positive_col = np.where(class_ids == 1)[0]
            if positive_col.size:
                positive_votes = votes[1:, positive_col[0]].astype(np.float32)
                total_votes = np.maximum(votes[1:, :].sum(axis=1).astype(np.float32), 1.0)
                return positive_votes / total_votes

    _, response = model.predict(samples)
    return (response.reshape(-1) > 0.5).astype(np.float32)


def tune_rtrees_threshold(
    model: cv2.ml_RTrees,
    pairs: list[tuple[str, Path, Path]],
    output_dir: Path,
    max_images: int = 0,
    use_region_growing: bool = True,
    grow_low_threshold: float = 0.26,
    grow_iterations: int = 12,
    slice_positions: Optional[dict[str, float]] = None,
) -> float:
    """自动寻找随机森林概率阈值。

    阈值越低，预测病灶越大，容易过分割；阈值越高，预测越保守，可能漏检。
    这里在一组阈值上试跑，并对预测面积明显大于标签的情况进行惩罚。
    """
    if not hasattr(model, "getVotes"):
        return 0.5

    thresholds = np.arange(0.50, 0.96, 0.03)
    scores = np.zeros_like(thresholds, dtype=np.float64)
    dice_scores = np.zeros_like(thresholds, dtype=np.float64)
    area_ratios = np.zeros_like(thresholds, dtype=np.float64)
    used = 0

    # 默认使用全部图像调阈值。若用户指定 max_images，则从全数据中均匀抽样，
    # 避免只看前几十张导致某些病灶阶段没有参与调参。
    if max_images and max_images < len(pairs):
        sample_indices = np.linspace(0, len(pairs) - 1, max_images).round().astype(int)
        subset = [pairs[int(i)] for i in sample_indices]
    else:
        subset = pairs
    total_pairs = len(subset)
    for index, (stem, image_path, mask_path) in enumerate(subset, start=1):
        progress_message("Tuning threshold", index, total_pairs, f"checking {image_path.name}")
        image, mask = load_pair(image_path, mask_path)
        roi = lung_roi_mask(image)
        slice_position = slice_positions.get(stem, 0.5) if slice_positions else None
        features, valid = image_features(image, roi, slice_position)
        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            continue

        probability = np.zeros(valid.shape, dtype=np.float32)
        batch_size = 60000
        for start in range(0, valid_idx.size, batch_size):
            idx = valid_idx[start : start + batch_size]
            probability[idx] = rtrees_probability(model, features[idx])
        probability_image = probability.reshape(image.shape)

        for threshold_index, threshold in enumerate(thresholds):
            prediction = (probability_image >= threshold).astype(np.uint8) * 255
            if use_region_growing:
                prediction = grow_prediction_from_seeds(
                    probability_image,
                    prediction,
                    roi,
                    grow_low_threshold,
                    grow_iterations,
                )
            prediction = postprocess_prediction(prediction, roi)
            pred_area = max(int((prediction > 0).sum()), 1)
            mask_area = max(int((mask > 0).sum()), 1)
            dice = dice_score(prediction, mask)
            area_ratio = pred_area / mask_area

            # 面积比 > 1 表示预测比标签大。这里对过分割惩罚更重，
            # 因为你目前遇到的主要问题就是病灶预测偏大。
            over_segmentation = max(0.0, area_ratio - 1.10)
            under_segmentation = max(0.0, 0.55 - area_ratio)
            scores[threshold_index] += dice - 0.22 * over_segmentation - 0.06 * under_segmentation
            dice_scores[threshold_index] += dice
            area_ratios[threshold_index] += area_ratio
        used += 1

    if used == 0:
        return 0.5
    mean_scores = scores / used
    mean_dice = dice_scores / used
    mean_area_ratio = area_ratios / used
    best_index = int(np.argmax(mean_scores))
    best_threshold = float(thresholds[best_index])

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = available_output_path(output_dir / "threshold_sweep.csv")
    with sweep_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "mean_dice",
                "mean_predicted_to_mask_area",
                "conservative_score",
                "p10_dice",
                "min_dice",
                "selection_score",
            ],
        )
        writer.writeheader()
        for threshold, dice, area_ratio, score in zip(thresholds, mean_dice, mean_area_ratio, mean_scores):
            writer.writerow(
                {
                    "threshold": f"{threshold:.2f}",
                    "mean_dice": f"{dice:.4f}",
                    "mean_predicted_to_mask_area": f"{area_ratio:.4f}",
                    "conservative_score": f"{score:.4f}",
                }
            )

    print(
        f"Best probability threshold: {best_threshold:.2f} "
        f"(mean tuning Dice {mean_dice[best_index]:.4f}, "
        f"pred/mask area {mean_area_ratio[best_index]:.2f})",
        flush=True,
    )
    print(f"Threshold sweep saved to: {sweep_path}", flush=True)
    return best_threshold


def estimate_lesion_area_ratio_limit(pairs: list[tuple[str, Path, Path]]) -> float:
    """根据标签估计病灶面积上限。

    如果模型把大片肺纹理都预测成病灶，Dice 会明显下降。
    因此用训练标签统计病灶面积占 ROI 的常见比例，作为后续保守限制。
    """
    ratios: list[float] = []
    total_pairs = len(pairs)
    for index, (_, image_path, mask_path) in enumerate(pairs, start=1):
        progress_message("Estimating lesion size prior", index, total_pairs, f"checking {image_path.name}")
        image, mask = load_pair(image_path, mask_path)
        roi = lung_roi_mask(image)
        roi_area = int((roi > 0).sum())
        lesion_area = int(((mask > 0) & (roi > 0)).sum())
        if roi_area > 0 and lesion_area > 0:
            ratios.append(lesion_area / roi_area)

    if not ratios:
        return 0.18
    limit = float(np.percentile(ratios, 80) * 1.15)
    return float(np.clip(limit, 0.02, 0.22))


def apply_area_limit(probability_image: np.ndarray, prediction: np.ndarray, roi: np.ndarray, area_ratio_limit: float) -> np.ndarray:
    """当预测面积过大时，只保留概率最高的一部分像素。"""
    if area_ratio_limit >= 0.99:
        return prediction

    roi_area = int((roi > 0).sum())
    max_area = max(20, int(roi_area * area_ratio_limit))
    pred_area = int((prediction > 0).sum())
    if pred_area <= max_area:
        return prediction

    candidate = (prediction > 0) & (roi > 0)
    values = probability_image[candidate]
    if values.size <= max_area:
        return prediction

    cutoff = np.partition(values, values.size - max_area)[values.size - max_area]
    limited = ((probability_image >= cutoff) & candidate).astype(np.uint8) * 255
    return limited


def grow_prediction_from_seeds(
    probability_image: np.ndarray,
    seed_prediction: np.ndarray,
    roi: np.ndarray,
    low_threshold: float,
    iterations: int,
) -> np.ndarray:
    """从高置信病灶核心向周围低置信候选区做区域生长。

    直接降低阈值会把所有低置信区域都变成病灶，容易产生孤立假阳性。
    区域生长只允许已经连接到病灶核心的候选区被纳入，适合扩张大片病灶边缘。
    """
    if iterations <= 0:
        return seed_prediction

    candidate = ((probability_image >= low_threshold) & (roi > 0)).astype(np.uint8) * 255
    grown = cv2.bitwise_and(seed_prediction, candidate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(iterations):
        expanded = cv2.dilate(grown, kernel, iterations=1)
        expanded = cv2.bitwise_and(expanded, candidate)
        if np.array_equal(expanded, grown):
            break
        grown = expanded
    return grown


def recover_undersegmented_prediction(
    probability_image: np.ndarray,
    seed_prediction: np.ndarray,
    roi: np.ndarray,
    threshold: float,
    threshold_delta: float,
    max_growth: float,
) -> np.ndarray:
    """Recover likely missed lesion border around existing high-confidence seeds.

    Lowest-Dice cases are often under-segmented: the model finds the lesion core
    but misses surrounding low-contrast lesion pixels. This step expands only
    from existing seeds into connected lower-confidence candidates, then caps
    the final area so it cannot flood the lung.
    """
    seed = cv2.bitwise_and(seed_prediction, seed_prediction, mask=roi)
    seed_area = int((seed > 0).sum())
    if seed_area == 0 or max_growth <= 1.0:
        return seed

    low_threshold = max(0.05, threshold - threshold_delta)
    candidate = ((probability_image >= low_threshold) & (roi > 0)).astype(np.uint8) * 255
    if int((candidate > 0).sum()) <= seed_area:
        return seed

    # Only allow candidates close to the original lesion core.
    near_seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    near_seed = cv2.dilate(seed, near_seed_kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    recovered = seed.copy()
    for label in range(1, num_labels):
        component = labels == label
        if not np.logical_and(component, near_seed > 0).any():
            continue
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 8:
            continue
        recovered[component] = 255

    max_area = max(seed_area, int(seed_area * max_growth))
    recovered_area = int((recovered > 0).sum())
    if recovered_area <= max_area:
        return cv2.bitwise_and(recovered, recovered, mask=roi)

    # If expansion is too large, keep the highest-probability recovered pixels.
    recovered_pixels = recovered > 0
    values = probability_image[recovered_pixels]
    if values.size <= max_area:
        return cv2.bitwise_and(recovered, recovered, mask=roi)

    cutoff = np.partition(values, values.size - max_area)[values.size - max_area]
    limited = ((probability_image >= cutoff) & recovered_pixels).astype(np.uint8) * 255
    return cv2.bitwise_and(limited, limited, mask=roi)


def segment_lesion_rtrees(
    image: np.ndarray,
    model: cv2.ml_RTrees,
    threshold: float,
    area_ratio_limit: float,
    use_region_growing: bool,
    grow_low_threshold: float,
    grow_iterations: int,
    slice_position: Optional[float] = None,
) -> np.ndarray:
    """使用随机森林模型对单张 CT 图像进行病灶分割。"""
    roi = lung_roi_mask(image)
    features, valid = image_features(image, roi, slice_position)
    probability = np.zeros(valid.shape, dtype=np.float32)

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return np.zeros_like(image, dtype=np.uint8)

    batch_size = 60000
    for start in range(0, valid_idx.size, batch_size):
        idx = valid_idx[start : start + batch_size]
        probability[idx] = rtrees_probability(model, features[idx])

    # 概率图经过阈值变成二值图；阈值越高，预测越保守。
    probability_image = probability.reshape(image.shape)
    prediction_image = (probability_image >= threshold).astype(np.uint8) * 255

    # 对大片低对比度病灶，随机森林常先识别出高置信核心，再漏掉边缘。
    # 区域生长可以在不引入远处孤立噪声的情况下补回这些边缘区域。
    if use_region_growing:
        prediction_image = grow_prediction_from_seeds(
            probability_image,
            prediction_image,
            roi,
            grow_low_threshold,
            grow_iterations,
        )

    # 如果预测区域明显过大，先按概率裁掉低置信区域，再做形态学后处理。
    prediction_image = apply_area_limit(probability_image, prediction_image, roi, area_ratio_limit)
    return postprocess_prediction(prediction_image, roi)


def extra_trees_probability_image(
    image: np.ndarray,
    model,
    slice_position: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ExtraTrees 对整张图的病灶概率图和肺部 ROI。"""
    roi = lung_roi_mask(image)
    features, valid = image_features(image, roi, slice_position)
    probability = np.zeros(valid.shape, dtype=np.float32)

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return probability.reshape(image.shape), roi

    batch_size = 60000
    for start in range(0, valid_idx.size, batch_size):
        idx = valid_idx[start : start + batch_size]
        probability[idx] = model.predict_proba(features[idx].astype(np.float32))[:, 1]

    return probability.reshape(image.shape), roi


def tune_extra_trees_threshold(
    model,
    pairs: list[tuple[str, Path, Path]],
    output_dir: Path,
    max_images: int = 0,
    use_region_growing: bool = False,
    grow_low_threshold: float = 0.26,
    grow_iterations: int = 12,
    use_undersegment_recovery: bool = True,
    recovery_threshold_delta: float = 0.18,
    recovery_max_growth: float = 1.4,
    slice_positions: Optional[dict[str, float]] = None,
) -> float:
    """为 ExtraTrees 自动选择概率阈值。"""
    thresholds = np.arange(0.30, 0.86, 0.03)
    scores = np.zeros_like(thresholds, dtype=np.float64)
    dice_scores = np.zeros_like(thresholds, dtype=np.float64)
    area_ratios = np.zeros_like(thresholds, dtype=np.float64)
    dice_by_threshold: list[list[float]] = [[] for _ in thresholds]
    used = 0

    if max_images and max_images < len(pairs):
        sample_indices = np.linspace(0, len(pairs) - 1, max_images).round().astype(int)
        subset = [pairs[int(i)] for i in sample_indices]
    else:
        subset = pairs

    total_pairs = len(subset)
    for index, (stem, image_path, mask_path) in enumerate(subset, start=1):
        progress_message("Tuning ExtraTrees threshold", index, total_pairs, f"checking {image_path.name}")
        image, mask = load_pair(image_path, mask_path)
        slice_position = slice_positions.get(stem, 0.5) if slice_positions else None
        probability_image, roi = extra_trees_probability_image(image, model, slice_position)

        for threshold_index, threshold in enumerate(thresholds):
            prediction = (probability_image >= threshold).astype(np.uint8) * 255
            if use_region_growing:
                prediction = grow_prediction_from_seeds(
                    probability_image,
                    prediction,
                    roi,
                    grow_low_threshold,
                    grow_iterations,
                )
            if use_undersegment_recovery:
                prediction = recover_undersegmented_prediction(
                    probability_image,
                    prediction,
                    roi,
                    float(threshold),
                    recovery_threshold_delta,
                    recovery_max_growth,
                )
            prediction = postprocess_prediction(prediction, roi)

            pred_area = max(int((prediction > 0).sum()), 1)
            mask_area = max(int((mask > 0).sum()), 1)
            dice = dice_score(prediction, mask)
            area_ratio = pred_area / mask_area
            over_segmentation = max(0.0, area_ratio - 1.15)
            under_segmentation = max(0.0, 0.55 - area_ratio)
            scores[threshold_index] += dice - 0.18 * over_segmentation - 0.05 * under_segmentation
            dice_scores[threshold_index] += dice
            area_ratios[threshold_index] += area_ratio
            dice_by_threshold[threshold_index].append(float(dice))
        used += 1

    if used == 0:
        return 0.5

    mean_scores = scores / used
    mean_dice = dice_scores / used
    mean_area_ratio = area_ratios / used
    p10_dice = np.array(
        [np.percentile(values, 10) if values else 0.0 for values in dice_by_threshold],
        dtype=np.float64,
    )
    min_dice = np.array([min(values) if values else 0.0 for values in dice_by_threshold], dtype=np.float64)

    # Optimize the average result, but also give weight to the low-Dice tail.
    # This is meant to avoid a nice average with a very poor worst-case group.
    selection_scores = 0.78 * mean_scores + 0.22 * p10_dice
    best_index = int(np.argmax(selection_scores))
    best_threshold = float(thresholds[best_index])

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = available_output_path(output_dir / "threshold_sweep.csv")
    with sweep_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "mean_dice",
                "mean_predicted_to_mask_area",
                "conservative_score",
                "p10_dice",
                "min_dice",
                "selection_score",
            ],
        )
        writer.writeheader()
        for threshold, dice, area_ratio, score, p10, min_value, select_score in zip(
            thresholds,
            mean_dice,
            mean_area_ratio,
            mean_scores,
            p10_dice,
            min_dice,
            selection_scores,
        ):
            writer.writerow(
                {
                    "threshold": f"{threshold:.2f}",
                    "mean_dice": f"{dice:.4f}",
                    "mean_predicted_to_mask_area": f"{area_ratio:.4f}",
                    "conservative_score": f"{score:.4f}",
                    "p10_dice": f"{p10:.4f}",
                    "min_dice": f"{min_value:.4f}",
                    "selection_score": f"{select_score:.4f}",
                }
            )

    print(
        f"Best ExtraTrees probability threshold: {best_threshold:.2f} "
        f"(mean tuning Dice {mean_dice[best_index]:.4f}, "
        f"pred/mask area {mean_area_ratio[best_index]:.2f})",
        flush=True,
    )
    print(f"Threshold sweep saved to: {sweep_path}", flush=True)
    return best_threshold


def segment_lesion_extra_trees(
    image: np.ndarray,
    model,
    threshold: float,
    area_ratio_limit: float,
    use_region_growing: bool,
    grow_low_threshold: float,
    grow_iterations: int,
    use_undersegment_recovery: bool,
    recovery_threshold_delta: float,
    recovery_max_growth: float,
    slice_position: Optional[float] = None,
) -> np.ndarray:
    """使用 ExtraTrees 模型对单张 CT 图像进行病灶分割。"""
    probability_image, roi = extra_trees_probability_image(image, model, slice_position)
    prediction_image = (probability_image >= threshold).astype(np.uint8) * 255

    if use_region_growing:
        prediction_image = grow_prediction_from_seeds(
            probability_image,
            prediction_image,
            roi,
            grow_low_threshold,
            grow_iterations,
        )

    if use_undersegment_recovery:
        prediction_image = recover_undersegmented_prediction(
            probability_image,
            prediction_image,
            roi,
            threshold,
            recovery_threshold_delta,
            recovery_max_growth,
        )

    prediction_image = apply_area_limit(probability_image, prediction_image, roi, area_ratio_limit)
    return postprocess_prediction(prediction_image, roi)


def split_train_test_pairs(
    pairs: list[tuple[str, Path, Path]],
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]], dict[str, str]]:
    """Split image-mask pairs into image-level train/test sets.

    The split is done by CT image, not by pixel. Test images are not used for
    model training or threshold tuning, so their Dice/IoU is a stricter estimate
    of generalization.
    """
    if test_ratio <= 0:
        split_map = {stem: "train" for stem, _, _ in pairs}
        return pairs, [], split_map

    if test_ratio >= 1:
        raise ValueError("--test-ratio must be smaller than 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    test_count = int(round(len(pairs) * test_ratio))
    test_count = min(max(test_count, 1), len(pairs) - 1)
    test_indices = set(indices[:test_count].tolist())

    train_pairs: list[tuple[str, Path, Path]] = []
    test_pairs: list[tuple[str, Path, Path]] = []
    split_map: dict[str, str] = {}

    for index, pair in enumerate(pairs):
        stem = pair[0]
        if index in test_indices:
            test_pairs.append(pair)
            split_map[stem] = "test"
        else:
            train_pairs.append(pair)
            split_map[stem] = "train"

    return train_pairs, test_pairs, split_map


def metric_summary(rows: list[dict[str, Union[str, float]]], name: str) -> str:
    """Build a short Markdown metric summary for one split."""
    if not rows:
        return f"- {name}图像数量：0\n"

    dice_values = np.array([float(row["dice"]) for row in rows])
    iou_values = np.array([float(row["iou"]) for row in rows])
    acc_values = np.array([float(row["pixel_accuracy"]) for row in rows])
    return (
        f"- {name}图像数量：{len(rows)}\n"
        f"- {name}平均 Dice：{dice_values.mean():.4f}\n"
        f"- {name}平均 IoU：{iou_values.mean():.4f}\n"
        f"- {name}平均像素准确率：{acc_values.mean():.4f}\n"
        f"- {name}最低 Dice：{dice_values.min():.4f}\n"
        f"- {name}最高 Dice：{dice_values.max():.4f}\n"
    )


def run(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    max_visuals: int,
    method: str,
    threshold: Optional[float],
    area_limit: Optional[float],
    tune_images: int,
    use_region_growing: bool,
    grow_low_threshold: float,
    grow_iterations: int,
    use_undersegment_recovery: bool,
    recovery_threshold_delta: float,
    recovery_max_growth: float,
    test_ratio: float,
    split_seed: int,
    use_slice_position: bool,
) -> None:
    """执行完整实验：配对数据、训练/预测、保存图像并汇总指标。"""
    images = collect_files(images_dir)
    masks = collect_files(masks_dir)
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    if not masks:
        raise ValueError(f"No masks found in {masks_dir}")

    pred_dir = output_dir / "predictions"
    overlay_dir = output_dir / "overlays"
    rows: list[dict[str, Union[str, float]]] = []
    pairs: list[tuple[str, Path, Path]] = []

    # 先把所有 image-mask 对配好，避免训练或预测到一半才发现缺文件。
    for stem, image_path in sorted(images.items()):
        mask_path = find_mask(stem, masks)
        if mask_path is None:
            print(f"[skip] No mask matched for {image_path.name}")
            continue
        pairs.append((stem, image_path, mask_path))

    if not pairs:
        raise ValueError("No image-mask pairs were matched. Please check file names.")

    train_pairs, test_pairs, split_map = split_train_test_pairs(pairs, test_ratio, split_seed)
    slice_positions = build_slice_positions(pairs) if use_slice_position else {}
    print(
        f"Dataset split: {len(train_pairs)} train images, {len(test_pairs)} test images "
        f"(test ratio {test_ratio:.2f}, seed {split_seed})",
        flush=True,
    )
    print(
        f"Slice position feature: {'enabled' if use_slice_position else 'disabled'}",
        flush=True,
    )

    model = None
    probability_threshold = 0.5
    lesion_area_ratio_limit = 1.0
    if method in ("rtrees", "extra_trees"):
        print("Training supervised pixel classifier from the provided masks...")
        if method == "extra_trees":
            model = train_extra_trees_classifier(train_pairs, slice_positions)
        else:
            model = train_rtrees_classifier(train_pairs, slice_positions)

        # 修复真实概率后，阈值已经能控制预测大小；默认不再启用硬面积上限。
        # 面积上限仍保留为手动参数，只有预测明显偏大时再使用。
        if area_limit is not None:
            lesion_area_ratio_limit = area_limit
            print(f"Manual lesion area ratio limit: {lesion_area_ratio_limit:.3f}", flush=True)
        else:
            print("Lesion area ratio limit: disabled", flush=True)

        # 如果用户没有手动指定阈值，就用标签自动调一个偏保守的阈值。
        if threshold is None:
            if method == "extra_trees":
                probability_threshold = tune_extra_trees_threshold(
                    model,
                    train_pairs,
                    output_dir,
                    max_images=tune_images,
                    use_region_growing=use_region_growing,
                    grow_low_threshold=grow_low_threshold,
                    grow_iterations=grow_iterations,
                    use_undersegment_recovery=use_undersegment_recovery,
                    recovery_threshold_delta=recovery_threshold_delta,
                    recovery_max_growth=recovery_max_growth,
                    slice_positions=slice_positions,
                )
            else:
                probability_threshold = tune_rtrees_threshold(
                    model,
                    train_pairs,
                    output_dir,
                    max_images=tune_images,
                    use_region_growing=use_region_growing,
                    grow_low_threshold=grow_low_threshold,
                    grow_iterations=grow_iterations,
                    slice_positions=slice_positions,
                )
        else:
            probability_threshold = threshold
            print(f"Using manual probability threshold: {probability_threshold:.2f}", flush=True)

    visual_count = 0
    total_pairs = len(pairs)
    for index, (stem, image_path, mask_path) in enumerate(pairs, start=1):
        progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: reading")
        image, mask = load_pair(image_path, mask_path)

        progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: predicting lesion")
        slice_position = slice_positions.get(stem, 0.5) if slice_positions else None
        if method == "rtrees":
            prediction = segment_lesion_rtrees(
                image,
                model,
                probability_threshold,
                lesion_area_ratio_limit,
                use_region_growing,
                grow_low_threshold,
                grow_iterations,
                slice_position,
            )
        elif method == "extra_trees":
            prediction = segment_lesion_extra_trees(
                image,
                model,
                probability_threshold,
                lesion_area_ratio_limit,
                use_region_growing,
                grow_low_threshold,
                grow_iterations,
                use_undersegment_recovery,
                recovery_threshold_delta,
                recovery_max_growth,
                slice_position,
            )
        else:
            prediction = segment_lesion(image)

        progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: saving prediction")
        write_image(pred_dir / f"{stem}_pred.png", prediction)

        # ROI 和 overlay 默认每张都保存；如果 --max-visuals > 0，则只保存前 N 张。
        if max_visuals <= 0 or visual_count < max_visuals:
            progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: saving ROI")
            roi = lung_roi_mask(image)
            write_image(output_dir / "lung_roi" / f"{stem}_roi.png", roi)

            progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: saving overlay")
            overlay = overlay_segmentation(image, prediction, mask, roi=roi)
            write_image(overlay_dir / f"{stem}_overlay.png", overlay)
            visual_count += 1

        progress_message("Segmenting + saving outputs", index, total_pairs, f"{stem}: calculating metrics")
        rows.append(
            {
                "split": split_map.get(stem, "unknown"),
                "image": image_path.name,
                "mask": mask_path.name,
                "dice": dice_score(prediction, mask),
                "iou": iou_score(prediction, mask),
                "pixel_accuracy": pixel_accuracy(prediction, mask),
                "predicted_area": int((prediction > 0).sum()),
                "mask_area": int((mask > 0).sum()),
            }
        )

    # 保存逐张指标，便于找出 Dice 最低的图像单独分析。
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = available_output_path(output_dir / "metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dice_values = np.array([float(row["dice"]) for row in rows])
    iou_values = np.array([float(row["iou"]) for row in rows])
    acc_values = np.array([float(row["pixel_accuracy"]) for row in rows])
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    split_summary = metric_summary(train_rows, "训练集") + "\n" + metric_summary(test_rows, "测试集")
    report = (
        "# 肺炎 CT 病灶分割实验结果\n\n"
        f"- 分割方法：{method}\n"
        f"- 随机森林概率阈值：{probability_threshold:.2f}\n"
        f"- 病灶面积比例上限：{'未启用' if lesion_area_ratio_limit >= 0.99 else f'{lesion_area_ratio_limit:.3f}'}\n"
        f"- 低置信区域生长：{'启用' if use_region_growing else '未启用'}\n"
        f"- 区域生长低阈值：{grow_low_threshold:.2f}\n"
        f"- 区域生长迭代次数：{grow_iterations}\n"
        f"- 欠分割恢复：{'启用' if use_undersegment_recovery else '未启用'}\n"
        f"- 欠分割恢复阈值差：{recovery_threshold_delta:.2f}\n"
        f"- 欠分割最大扩张倍数：{recovery_max_growth:.2f}\n"
        f"- 测试集比例：{test_ratio:.2f}\n"
        f"- 划分随机种子：{split_seed}\n"
        f"- 切片位置特征：{'启用' if use_slice_position else '未启用'}\n"
        f"- 图像数量：{len(rows)}\n"
        f"- 平均 Dice：{dice_values.mean():.4f}\n"
        f"- 平均 IoU：{iou_values.mean():.4f}\n"
        f"- 平均像素准确率：{acc_values.mean():.4f}\n"
        f"- 最低 Dice：{dice_values.min():.4f}\n"
        f"- 最高 Dice：{dice_values.max():.4f}\n\n"
        "## 训练/测试集结果\n\n"
        f"{split_summary}\n"
        "## 方法说明\n\n"
        "本实验先基于肺部低灰度特征提取肺野 ROI，再在 ROI 内进行病灶分割。"
        "若使用 extra_trees 或 rtrees 方法，程序会利用提供的标签抽样训练随机森林类像素分类器，"
        "学习灰度、局部均值、局部纹理、梯度、空间位置、到肺边界距离等特征；"
        "如果启用切片位置特征，模型还会利用连续 CT 切片的上下位置作为额外信息。"
        "若使用 threshold 方法，则采用 CLAHE、Otsu 阈值、形态学处理和连通域筛选。"
        "前景为 ROI 内预测病灶区域，背景为非病灶区域。\n\n"
        "## 可视化说明\n\n"
        "overlay 图中红色为算法预测区域，绿色为人工标签区域，黄色为二者重叠区域。"
        "黄色越多且红绿单独区域越少，说明分割效果越好。\n"
    )
    report_path = available_output_path(output_dir / "analysis_report.md")
    report_path.write_text(report, encoding="utf-8")

    print(f"Finished {len(rows)} image-mask pairs.")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print(f"Overlays: {overlay_dir}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便直接调方法、阈值和输出数量。"""
    parser = argparse.ArgumentParser(description="Pneumonia CT lesion segmentation experiment")
    # 调参总原则：
    # 1. 预测病灶偏大：提高 --threshold，例如 0.65 -> 0.70；仍偏大时再加 --area-limit。
    # 2. 预测病灶偏小：降低 --threshold，例如 0.65 -> 0.60；不要设置过小的 --area-limit。
    # 3. 跑得太慢：调试阶段可以设置 --tune-images 40 或 60；最终实验建议用默认 0。
    # 4. 可视化文件太多：设置 --max-visuals 20，只保存前 20 张 ROI/overlay。
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"folder containing CT images, default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=DEFAULT_MASKS_DIR,
        help=f"folder containing lesion masks, default: {DEFAULT_MASKS_DIR}",
    )
    # 输出目录。建议每次调参换一个目录名，方便比较不同实验结果。
    # 例：--output results_t065、--output results_no_area_limit。
    parser.add_argument(
        "--output",
        type=Path,
        default=PYCHARM_OUTPUT_DIR,
        help="output folder; change it to keep results from different parameter trials",
    )
    parser.add_argument(
        "--max-visuals",
        type=int,
        default=PYCHARM_MAX_VISUALS,
        # 只影响 ROI 和 overlay 的保存数量，不影响 predictions/metrics。
        # 0 = 全部保存；20 = 只保存前 20 张，适合快速调试。
        help="number of ROI/overlay images to save; 0 means save all images; use 20 for quick checks",
    )
    parser.add_argument(
        "--method",
        choices=("extra_trees", "rtrees", "threshold"),
        default=PYCHARM_METHOD,
        # extra_trees：sklearn 极端随机森林，通常拟合能力更强。
        # rtrees：OpenCV 随机森林，速度较快。
        # threshold：传统阈值法 baseline，适合写报告时做方法对比。
        help="segmentation method; extra_trees is stronger, rtrees is faster, threshold is baseline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PYCHARM_THRESHOLD,
        # 随机森林概率阈值，是最常用的调参旋钮。
        # 阈值越高，预测区域越小、越保守；阈值越低，预测区域越大、召回更多。
        # 预测偏大：试 0.65、0.70、0.75。
        # 预测偏小：试 0.55、0.60。
        # 不填时会自动扫描阈值，并生成 threshold_sweep.csv。
        help="manual rtrees probability threshold; higher makes prediction smaller; default is automatic tuning",
    )
    parser.add_argument(
        "--area-limit",
        type=float,
        default=PYCHARM_AREA_LIMIT,
        # 病灶面积上限，占肺部 ROI 的比例。默认不启用。
        # 只有预测明显偏大时才建议使用，例如 0.10-0.18。
        # 如果真实病灶本来很大，area-limit 太小会把病灶截断，Dice 会下降。
        help="manual lesion area limit relative to lung ROI, e.g. 0.12; default is disabled",
    )
    parser.add_argument(
        "--tune-images",
        type=int,
        default=PYCHARM_TUNE_IMAGES,
        # 自动调阈值时使用多少张图。
        # 0 = 使用全部 200 张，最稳但较慢；40/60 = 更快，适合调试。
        help="number of images used for automatic threshold tuning; 0 means use all paired images",
    )
    parser.add_argument(
        "--no-region-growing",
        action="store_true",
        default=not PYCHARM_USE_REGION_GROWING,
        # 关闭低置信区域生长。一般不建议关闭，除非你发现预测明显扩太大。
        help="disable low-confidence region growing",
    )
    parser.add_argument(
        "--grow-low-threshold",
        type=float,
        default=PYCHARM_GROW_LOW_THRESHOLD,
        # 区域生长候选阈值。越低扩张越多；越高越保守。
        help="low probability threshold used by region growing; lower grows more",
    )
    parser.add_argument(
        "--grow-iterations",
        type=int,
        default=PYCHARM_GROW_ITERATIONS,
        # 区域生长次数。越大扩张越远；太大可能贴到无关纹理。
        help="number of region-growing dilation iterations",
    )
    parser.add_argument(
        "--no-undersegment-recovery",
        action="store_true",
        default=not PYCHARM_USE_UNDERSEGMENT_RECOVERY,
        # 关闭欠分割恢复。若你发现预测明显偏大，可以关闭或降低 recovery max growth。
        help="disable under-segmentation recovery",
    )
    parser.add_argument(
        "--recovery-threshold-delta",
        type=float,
        default=PYCHARM_RECOVERY_THRESHOLD_DELTA,
        # 恢复候选阈值 = 主阈值 - delta。delta 越大，补回越多低置信边缘。
        help="threshold delta used by under-segmentation recovery",
    )
    parser.add_argument(
        "--recovery-max-growth",
        type=float,
        default=PYCHARM_RECOVERY_MAX_GROWTH,
        # 恢复后预测面积最多是原始预测面积的多少倍。
        help="maximum recovered area relative to original prediction",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=PYCHARM_TEST_RATIO,
        # 图像级测试集比例。0.2 表示 8:2 划分，测试图像不参与训练和阈值调优。
        help="image-level test split ratio; 0.2 means 80% train and 20% test",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=PYCHARM_SPLIT_SEED,
        # 固定随机种子，保证每次 PyCharm 运行得到同一组训练/测试划分。
        help="random seed for reproducible train/test split",
    )
    parser.add_argument(
        "--no-slice-position",
        action="store_true",
        default=not PYCHARM_USE_SLICE_POSITION,
        # 如果 200 张图像是同一套连续 CT 切片，保留切片位置特征通常能提升测试集 Dice。
        # 如果数据来自不同病人或不同扫描序列，请关闭它，避免模型记住文件编号规律。
        help="disable slice-order feature; use this if images are not consecutive CT slices",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.images,
        args.masks,
        args.output,
        args.max_visuals,
        args.method,
        args.threshold,
        args.area_limit,
        args.tune_images,
        not args.no_region_growing,
        args.grow_low_threshold,
        args.grow_iterations,
        not args.no_undersegment_recovery,
        args.recovery_threshold_delta,
        args.recovery_max_growth,
        args.test_ratio,
        args.split_seed,
        not args.no_slice_position,
    )
