# IRP-Net
## 工程文件树

```
├── checkpoints
│   ├── checkpoints
│   │   ├── best.pth
│   │   └── last.pth
│   └── train_log.csv
├── data_template
│   ├── dataset_experiment
│   │   ├── 001.png
│   │   ├── 2-40.jpg
│   │   ├── img1.png
│   │   └── img2.png
│   └── dataset_template
│       ├── images
│       │   ├── Mask
│       │   │   ├── 1_1.422.png
│       │   │   ├── 1_1.781.png
│       │   │   ├── 1_2.141.png
│       │   │   ├── 1_2.500.png
│       │   │   ├── 1_2.860.png
│       │   │   ├── 1_3.219.png
│       │   │   ├── 1_3.579.png
│       │   │   ├── 1_3.939.png
│       │   │   ├── 1_4.298.png
│       │   │   └── 1_4.658.png
│       │   └── ROI
│       │       ├── 1_1.422.png
│       │       ├── 1_1.781.png
│       │       ├── 1_2.141.png
│       │       ├── 1_2.500.png
│       │       ├── 1_2.860.png
│       │       ├── 1_3.219.png
│       │       ├── 1_3.579.png
│       │       ├── 1_3.939.png
│       │       ├── 1_4.298.png
│       │       └── 1_4.658.png
│       └── labels
│           ├── Mask
│           │   ├── 1_1.422.png
│           │   ├── 1_1.781.png
│           │   ├── 1_2.141.png
│           │   ├── 1_2.500.png
│           │   ├── 1_2.860.png
│           │   ├── 1_3.219.png
│           │   ├── 1_3.579.png
│           │   ├── 1_3.939.png
│           │   ├── 1_4.298.png
│           │   └── 1_4.658.png
│           └── Phi_w
│               ├── 1_1.422.png
│               ├── 1_1.781.png
│               ├── 1_2.141.png
│               ├── 1_2.500.png
│               ├── 1_2.860.png
│               ├── 1_3.219.png
│               ├── 1_3.579.png
│               ├── 1_3.939.png
│               ├── 1_4.298.png
│               └── 1_4.658.png
├── deploy
│   ├── export.py
│   └── test_deploy.py
├── models
│   ├── __init__.py
│   ├── model_CBAM_ResUNet.py
│   ├── model_CBAM_ResUNet_nobridge.py
│   ├── model_CBAM_UNet.py
│   ├── model_CBAM_UNet_nobridge.py
│   ├── model_ResUNet.py
│   └── model_UNet.py
├── results
│   └── CBAM_ResUNet_nobridge_260205144124_prediction_result.png
├── scripts
│   ├── handle_checkpoints.py
│   └── visualize_cbam_unet.py
├── utils
│   ├── accuracy.py
│   ├── builder.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── loss.py
│   └── trainer.py
├── config.yaml
├── predict.py
└── train.py
```

## 数据集设计
### 数据集组成
- 特征集：
    1. images文件夹中ROI目录下的图像文件（256x256灰度图），
    2. images文件夹中Mask目录下的图像文件（256x256灰度图或256x256大小的二值图），
    - 将上述两个路径中的同名文件从通道维度合并为一个输入样本。ROI放在通道0，Mask放在通道1。
- 标签集：
    1. labels文件夹中Mask目录下的图像文件（256x256灰度图或256x256大小的二值图）,
    2. labels文件夹中Phi_w目录下的图像文件(包裹相位,256x256灰度图),
    - 将Phi_w目录下包裹相位图映射到[-pi,pi]范围,分别求其对应的sin和cos矩阵
    将上述两个路径中的同名文件对应的Mask,sin矩阵,cos矩阵从通道维度合并为一个标签样本。Mask放在通道0，sin矩阵放在通道1，cos矩阵放在通道2。

### 数据加载器配置
- 数据加载器根据config.yaml文件中imgs_dir，labels_dir的路径参数加载数据集，按splits的参数[train, val, test]比例加载不同的数据集子集   
- 数据加载器根据config.yaml文件中batch_size参数设置批量加载数据
- 数据增强配置
    - 数据集打乱样本顺序
    - 特征集与标签集保持一致的增强操作包括：数据增强包括随机裁剪、随机水平翻转、随机垂直翻转、随机旋转等操作，对于随机旋转等操作导致出现的边界问题，采用补0的方式处理。
    - 只对特征集进行增强操作，标签集保持原始样本的操作包括：对特征集中的ROI图像进行随机对比度调整
    - 数据增强配置包括是否开启样本打乱，是否开启数据增强、随机裁剪的大小、随机旋转的角度范围、随机对比度调整的范围等，这些参数从config.yaml文件中读取。
    >注：数据增强后需要调整缩放（对于Mask采用最近邻插值，其它用双线性插值），确保特征集与标签集中的所有矩阵大小依然保持256x256的长宽大小，Mask矩阵保持二值化状态''

## 损失函数设计
### 掩码加权损失
