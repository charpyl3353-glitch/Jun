# 肺炎 CT 分割参数调节说明

运行命令示例：

```powershell
python pneumonia_ct_segmentation.py
```

## 如果你用 PyCharm 运行

如果你不是在命令行里运行，而是在 PyCharm 里点绿色运行按钮，最简单的方法是直接修改 `pneumonia_ct_segmentation.py` 顶部的“PyCharm 调参区”：

```python
PYCHARM_THRESHOLD = None
PYCHARM_AREA_LIMIT = None
PYCHARM_TUNE_IMAGES = 0
PYCHARM_MAX_VISUALS = 0
PYCHARM_OUTPUT_DIR = Path("results")
PYCHARM_METHOD = "extra_trees"
PYCHARM_TEST_RATIO = 0.2
PYCHARM_SPLIT_SEED = 2026
PYCHARM_USE_UNDERSEGMENT_RECOVERY = True
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.18
PYCHARM_RECOVERY_MAX_GROWTH = 1.4
PYCHARM_USE_SLICE_POSITION = True
```

当前推荐最终方法是：

```python
PYCHARM_METHOD = "extra_trees"
PYCHARM_THRESHOLD = None
PYCHARM_AREA_LIMIT = None
PYCHARM_USE_REGION_GROWING = False
```

这组参数会使用 sklearn 的 ExtraTrees 像素分类器，并自动搜索概率阈值。当前结果中，全体平均 Dice 约 `0.8821`，测试集平均 Dice 约 `0.8177`。

如果这 200 张图像是同一套 CT 的连续切片，建议保留：

```python
PYCHARM_USE_SLICE_POSITION = True
```

它会把“第几张切片”作为一个额外特征，让模型利用相邻 CT 切片病灶位置连续变化的规律。当前 8:2 划分下，新结果测试集平均 Dice 约为 `0.8177`。

如果你的图像来自不同病人或不同扫描序列，不要使用这个特征，应改成：

```python
PYCHARM_USE_SLICE_POSITION = False
```

如果最低 Dice 偏低且 `predicted_area` 明显小于 `mask_area`，说明低分图主要是漏分割。当前脚本默认启用欠分割恢复：

```python
PYCHARM_USE_UNDERSEGMENT_RECOVERY = True
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.18
PYCHARM_RECOVERY_MAX_GROWTH = 1.4
```

它会从高置信病灶核心向周围低置信但相邻的区域补回一部分边缘，并限制最大扩张面积，避免整片肺被误分割。

现在默认启用 8:2 图像级训练/测试划分：

```python
PYCHARM_TEST_RATIO = 0.2
```

含义是：

- 80% CT 图像用于训练和自动阈值调优。
- 20% CT 图像完全不参与训练，只用于最终测试。
- `metrics.csv` 中的 `split` 列会标出每张图属于 `train` 还是 `test`。
- `analysis_report.md` 会分别汇总训练集和测试集 Dice/IoU。

常见改法：

```python
# 预测偏大时
PYCHARM_THRESHOLD = 0.70
PYCHARM_OUTPUT_DIR = Path("results_t070")
```

```python
# 预测仍然偏大时
PYCHARM_THRESHOLD = 0.70
PYCHARM_AREA_LIMIT = 0.15
PYCHARM_OUTPUT_DIR = Path("results_t070_a015")
```

```python
# 预测偏小时
PYCHARM_THRESHOLD = 0.60
PYCHARM_AREA_LIMIT = None
PYCHARM_OUTPUT_DIR = Path("results_t060")
```

如果你熟悉 PyCharm 的 Run Configuration，也可以在 `Parameters` 里填写命令行参数，例如：

```text
--threshold 0.70 --output results_t070
```

但对本实验来说，直接改顶部“PyCharm 调参区”更方便。

## 常用参数

### `--method`

选择分割方法。

- `extra_trees`：推荐方法，sklearn 极端随机森林，效果最好。
- `rtrees`：OpenCV 随机森林，速度较快，但效果略低。
- `threshold`：传统阈值法 baseline，用于实验对比。

PyCharm 中对应：

```python
PYCHARM_METHOD = "extra_trees"
```

### `--threshold`

随机森林输出的病灶概率阈值，是最常用的调参参数。

- 阈值越高：预测越保守，病灶区域越小。
- 阈值越低：预测越宽松，病灶区域越大。

适用情况：

- 预测病灶比真实标签大很多：提高阈值。
- 预测病灶比真实标签小很多：降低阈值。

示例：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.70 --output results_t070
```

建议尝试：

```text
0.60, 0.65, 0.70, 0.75
```

### `--area-limit`

限制预测病灶面积占肺部 ROI 的最大比例，默认不启用。

- 预测明显偏大时可以使用。
- 真实病灶很大时不要设太小，否则会把病灶截断，Dice 反而下降。

示例：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.70 --area-limit 0.15 --output results_t070_a015
```

建议尝试：

```text
0.10, 0.12, 0.15, 0.18
```

### 欠分割恢复参数

这些参数主要用于优化最低 Dice，尤其是 `predicted_area` 明显小于 `mask_area` 的图像。

- `PYCHARM_USE_UNDERSEGMENT_RECOVERY`：是否启用欠分割恢复。
- `PYCHARM_RECOVERY_THRESHOLD_DELTA`：恢复候选阈值差。越大，补回越多低置信区域。
- `PYCHARM_RECOVERY_MAX_GROWTH`：恢复后面积最多是原始预测面积的多少倍。

建议：

```python
# 默认，比较均衡
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.18
PYCHARM_RECOVERY_MAX_GROWTH = 1.4

# 如果低分图还是明显预测偏小
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.22
PYCHARM_RECOVERY_MAX_GROWTH = 1.8

# 如果恢复后预测偏大
PYCHARM_RECOVERY_THRESHOLD_DELTA = 0.10
PYCHARM_RECOVERY_MAX_GROWTH = 1.2
```

### `--tune-images`

自动调阈值时使用多少张图像。

- `0`：使用全部图像，结果更稳，但运行更慢。
- `40` 或 `60`：只抽样部分图像，适合快速调试。

示例：

```powershell
python pneumonia_ct_segmentation.py --tune-images 60 --output results_fast_tune
```

最终写实验结果时，建议使用默认 `0`。

### `--test-ratio`

图像级测试集比例。

- `0.2`：8:2 划分，推荐用于更严谨的实验评价。
- `0`：不划分测试集，全部图像训练并评价，结果会偏乐观。

PyCharm 中对应：

```python
PYCHARM_TEST_RATIO = 0.2
```

### `--split-seed`

训练/测试划分随机种子。固定这个值可以保证每次运行划分一致。

PyCharm 中对应：

```python
PYCHARM_SPLIT_SEED = 2026
```

### `--max-visuals`

控制保存多少张 `lung_roi` 和 `overlay` 可视化图。

- `0`：全部保存。
- `20`：只保存前 20 张，适合快速检查。

这个参数不影响 `predictions` 和 `metrics.csv`。

示例：

```powershell
python pneumonia_ct_segmentation.py --max-visuals 20
```

### `--output`

指定输出文件夹。每次调参建议换一个输出目录，方便对比。

示例：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.70 --output results_t070
```

## 推荐调参顺序

1. 先使用推荐默认设置运行一次。

```powershell
python pneumonia_ct_segmentation.py --output results
```

2. 打开 `results_default/threshold_sweep.csv`。

重点看：

- `mean_dice`：平均 Dice，越高越好。
- `mean_predicted_to_mask_area`：预测面积 / 真实标签面积，越接近 `1.0` 越好。

3. 如果预测偏大，试：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.70 --output results_t070
```

4. 如果还是偏大，试：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.70 --area-limit 0.15 --output results_t070_a015
```

5. 如果预测偏小，试：

```powershell
python pneumonia_ct_segmentation.py --threshold 0.60 --output results_t060
```

## 结果判断

如果出现：

```text
predicted_area 远大于 mask_area
```

说明预测偏大，应提高 `--threshold`，必要时加 `--area-limit`。

如果出现：

```text
predicted_area 远小于 mask_area
```

说明预测偏小，应降低 `--threshold`，并取消或调大 `--area-limit`。

如果 Dice 低但像素准确率很高，不代表分割好。肺部 CT 图像背景像素很多，背景分对了也会让像素准确率很高；病灶分割主要看 Dice 和 IoU。

## 当前较优结果

使用 `extra_trees` 方法、自动阈值、不开启面积限制和区域生长，并启用连续切片位置特征时，本数据集 8:2 划分实测结果为：

```text
训练集平均 Dice：0.8982
测试集平均 Dice：0.8177
测试集平均 IoU：0.6963
测试集最低 Dice：0.6523
测试集最高 Dice：0.9109
```

这说明 `extra_trees + 切片位置特征` 明显优于之前的 `rtrees` 和纯 2D ExtraTrees，更适合作为当前数据集的最终实验结果。
