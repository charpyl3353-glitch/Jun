# 肺炎 CT 图像分割实验

本项目用于完成“医学成像与图像处理”中的肺炎 CT 病灶分割实验。程序会把病灶区域作为前景、非病灶区域作为背景，并使用提供的标签评价分割效果。

## 使用方法

先安装依赖：

```powershell
pip install opencv-python numpy
```

运行实验：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py
```

脚本已经内置了本次实验的默认图像路径和标签路径。默认方法为 `rtrees`，会利用标签训练随机森林像素分类器，再输出分割结果。若你的数据放在其他位置，再使用下面这种完整写法：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py `
  --images "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/images 200" `
  --masks "D:/HuaweiMoveData/Users/a2710/Desktop/生医大三下/医学成像与图像处理/图像处理部分实验/masks 200" `
  --output results
```

如果想运行原来的传统阈值分割 baseline：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py --method threshold
```

如果预测病灶明显偏大，可以手动提高随机森林概率阈值：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py --threshold 0.65
```

阈值越高，预测区域通常越小。程序默认会使用全部图像自动调阈值；如果想加快调参，可以只均匀抽样一部分图像：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py --tune-images 60
```

如果预测仍然明显偏大，可以手动限制预测病灶面积占肺部 ROI 的比例：

```powershell
& "D:/ProgramData/anaconda3/envs/2025ai/python.exe" pneumonia_ct_segmentation.py --threshold 0.65 --area-limit 0.12
```

面积上限默认不启用，因为部分图像真实病灶面积很大，硬限制会导致漏分割。自动调阈值时，程序会生成 `results/threshold_sweep.csv`，里面记录不同阈值下的平均 Dice 和预测/真实面积比。若预测偏大，优先选择面积比更接近 1 的较高阈值。

运行后会生成：

- `results/predictions/`：每张 CT 图像的预测病灶二值图。
- `results/lung_roi/`：每张 CT 图像的肺部 ROI 二值图，用于观察肺野提取效果。
- `results/overlays/`：每张 CT 图像的分割结果叠加图，红色为预测，绿色为标签，黄色为重叠。
- `results/metrics.csv`：每张图的 Dice、IoU、像素准确率等指标。
- `results/threshold_sweep.csv`：自动阈值搜索结果，用于观察阈值、Dice 和预测面积之间的关系。
- `results/analysis_report.md`：自动汇总的实验结果。

运行时会显示两个主要进度：

- `Training classifier`：训练监督像素分类器。
- `Segmenting + saving outputs`：逐张生成预测图、ROI 图、overlay 图并计算指标。

普通文字进度会显示当前图像正在执行的步骤，例如 `saving ROI`、`saving overlay`，不需要额外安装进度条库。

如果只想抽样保存一部分可视化图，可以使用：

```powershell
python pneumonia_ct_segmentation.py --max-visuals 20
```

## 算法原理

实验默认采用“肺部 ROI + 随机森林像素分类”的监督分割方法，适合利用本实验提供的病灶标签来提升 Dice。流程会先提取肺部 ROI，再只在肺部区域内寻找病灶，减少胸壁、床板、图像背景等区域的干扰：

1. 灰度归一化：减少不同 CT 图像亮度范围差异。
2. 肺部 ROI 提取：先提取患者身体区域，排除体外黑色背景；再在身体内部根据肺野低灰度特征寻找肺部候选区，并分别在左右半肺区域筛选候选连通域；同时去除靠近横膈膜方向的底部横向大块，避免把肺外区域识别成肺。对于肺底层面，允许 ROI 面积较小，避免用人工椭圆替代真实肺野。
3. ROI 裁剪与约束：根据肺部 ROI 的外接矩形裁剪图像，并在后续步骤中只保留 ROI 内的分割结果。
4. 特征提取：对 ROI 内像素提取灰度、CLAHE 增强灰度、局部均值、局部标准差、梯度、空间坐标和到肺边界距离等特征。
5. 随机森林训练：从人工标签中抽样病灶像素和更多背景像素，训练前景/背景分类器，减少过分割。
6. 阈值自动调优：用训练图像扫描不同概率阈值，并惩罚明显大于标签的预测结果，避免把大片肺纹理误分成病灶。
8. 像素分类预测：对每张 CT 的肺部 ROI 像素预测是否属于病灶。
9. 可选面积限制：若手动设置 `--area-limit`，则在预测面积过大时只保留模型最有把握的高概率区域。
10. 形态学开闭运算：去除小噪声并填补病灶内部空洞。
11. 线状伪影过滤：根据连通域长宽比、位置和填充率去除扫描床、胸壁边缘等细长亮线。
12. 肺边界轮廓过滤：去除紧贴 ROI 边缘、形态细长的肺边界伪影。
13. 连通域筛选：保留面积较大的疑似病灶区域，减少孤立噪声。

程序也保留了传统阈值方法，可通过 `--method threshold` 运行，用于和监督分类方法进行对比。

## 评价指标

- Dice：衡量预测区域和人工标签的重叠程度，越接近 1 越好。
- IoU：交并比，反映预测和标签交集占并集的比例。
- Pixel Accuracy：逐像素分类准确率。

## 结果分析写法参考

可以根据 `results/analysis_report.md` 中的平均 Dice、平均 IoU 和可视化图进行分析：

加入肺部 ROI 后，算法会先排除肺外背景和胸壁区域，再在肺部内部寻找病灶候选区域，因此通常能减少肺外误分割。ROI 提取时分别处理左右半肺，避免只保留一侧肺；同时限制横膈膜方向的底部区域，减少把肺外组织误当成肺。与单纯阈值法相比，随机森林能够利用标签学习病灶的灰度和纹理特征，对磨玻璃影、边界不清晰病灶通常更稳定。Dice 和 IoU 越高，说明算法预测与标签重叠越充分。若某些图像指标较低，可结合 ROI 图和 overlay 图观察是 ROI 提取不完整、病灶灰度接近正常肺组织、边界不完整还是小病灶遗漏造成的。

若预测图下方出现弧形或横向白线，通常不是肺炎病灶，而是 CT 扫描床、胸壁或肺外边界结构被阈值误识别。当前程序已经加入线状伪影过滤步骤，用于在后处理中去除这类细长结构。

若预测图中出现贴着肺部边缘的奇怪弧形轮廓，通常是肺边界或胸膜附近高亮结构被误分割。当前程序加入了肺边界轮廓过滤，会优先删除紧贴 ROI 边缘且形态细长的连通域。

## 可改进方向

- 尝试区域生长、K-means、GrabCut 或分水岭算法，与当前方法比较。
- 根据标签统计病灶灰度范围，调节 ROI 内的阈值百分位数。
- 若允许使用深度学习，可用 U-Net 训练监督分割模型，一般能获得更稳定的病灶边界。
