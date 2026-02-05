# data_loader.py
import os
import glob
from typing import List, Tuple, Dict

import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import math


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


class SegmentationDataset(Dataset):
    """
    灰度图像语义分割数据集：
      - image: [1, H, W], float32, 0~1
      - label:  [H, W],    long (类别索引，适合 CrossEntropyLoss)
    """
    def __init__(
        self,
        img_paths: List[str],
        label_paths: List[str],
        img_height: int = 256,
        img_width: int = 256,
        augment: bool = False,
    ):
        assert len(img_paths) == len(label_paths), "img_paths 和 label_paths 数量不一致"
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_image(self, path: str) -> Image.Image:
        # 转成灰度图 L
        img = Image.open(path).convert("L")
        # 保证大小正确（你原本就处理成 256x256 的话，这里可以去掉 resize）
        if img.size != (self.img_width, self.img_height):
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        return img

    def _load_label(self, path: str) -> Image.Image:
        # 同样以灰度读入，作为 label
        label = Image.open(path).convert("L")
        if label.size != (self.img_width, self.img_height):
            label = label.resize((self.img_width, self.img_height), Image.NEAREST)
        return label

    def _random_augment(
        self, img: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # 随机水平翻转
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机垂直翻转
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        # 随机旋转 0/90/180/270
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            angle = 90 * k
            img = img.rotate(angle)
            label = label.rotate(angle)

        return img, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        img = self._load_image(img_path)
        label = self._load_label(label_path)

        # 数据增强
        if self.augment:
            img, label = self._random_augment(img, label)

        # 转 tensor
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W], 0~1
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        # label 作为类别索引：0/1/...，这里直接取灰度值
        # 如果你的标签是 0/255 二值，可以先除以 255 再转 long
        phi_gt = np.array(label, dtype=np.float32) / 255.0 * 2 * math.pi - math.pi
        cos_gt = torch.cos(torch.from_numpy(phi_gt))
        sin_gt = torch.sin(torch.from_numpy(phi_gt))

        gt_cs  = torch.cat([cos_gt.unsqueeze(0), sin_gt.unsqueeze(0)], dim=0) # [2, H, W]


        return {"image": img_tensor, "label": gt_cs}


def load_data_paths(
    imgs_dir: str, labels_dir: str
) -> Tuple[List[str], List[str]]:
    """
    根据 imgs_dir 中的文件名在 labels_dir 中寻找同名标签。
    """
    img_files = [
        f for f in sorted(glob.glob(os.path.join(imgs_dir, "*")))
        if is_image_file(f)
    ]
    if len(img_files) == 0:
        raise RuntimeError(f"在 {imgs_dir} 中没有找到图像文件")

    img_paths = []
    label_paths = []

    for img_path in img_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, basename)
        if not os.path.isfile(label_path):
            raise RuntimeError(f"标签文件不存在: {label_path}")
        img_paths.append(img_path)
        label_paths.append(label_path)

    return img_paths, label_paths


def split_dataset(
    img_paths: List[str],
    label_paths: List[str],
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
]:
    """
    按比例随机划分训练/验证/测试集。
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits 之和必须为 1"
    assert len(img_paths) == len(label_paths)

    n = len(img_paths)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    # 剩下的给 test
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def select(idxs):
        return [img_paths[i] for i in idxs], [label_paths[i] for i in idxs]

    train_imgs, train_labels = select(train_idx)
    val_imgs, val_labels = select(val_idx)
    test_imgs, test_labels = select(test_idx)

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


def create_dataloaders_from_yaml(
    config_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    从 YAML 配置创建 train/val/test dataloader。

    YAML 示例：
    data:
      imgs_dir: data/raw/imgs
      labels_dir: data/raw/labels
      splits: [0.7, 0.2, 0.1]
      img_height: 256
      img_width: 256
      channels: 1
      augment: true
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    imgs_dir = data_cfg["imgs_dir"]
    labels_dir = data_cfg["labels_dir"]
    splits = tuple(data_cfg.get("splits", [0.7, 0.2, 0.1]))
    img_height = int(data_cfg.get("img_height", 256))
    img_width = int(data_cfg.get("img_width", 256))
    channels = int(data_cfg.get("channels", 1))
    augment = bool(data_cfg.get("augment", True))

    if channels != 1:
        raise ValueError(
            f"当前实现仅支持灰度图 (channels=1)，但配置中 channels={channels}"
        )

    img_paths, label_paths = load_data_paths(imgs_dir, labels_dir)

    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = \
        split_dataset(img_paths, label_paths, splits=splits, seed=seed)

    train_dataset = SegmentationDataset(
        train_imgs, train_labels,
        img_height=img_height,
        img_width=img_width,
        augment=augment,
    )
    val_dataset = SegmentationDataset(
        val_imgs, val_labels,
        img_height=img_height,
        img_width=img_width,
        augment=False,  # 验证和测试一般不做增强
    )
    test_dataset = SegmentationDataset(
        test_imgs, test_labels,
        img_height=img_height,
        img_width=img_width,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    # 简单测试一下加载是否正常
    loaders = create_dataloaders_from_yaml(
        config_path="config.yaml",
        batch_size=4,
        num_workers=0,
    )
    batch = next(iter(loaders["train"]))
    print("image shape:", batch["image"].shape)  # [B, 1, 256, 256]
    print("label shape:", batch["label"].shape)    # [B, 1, 256, 256]
