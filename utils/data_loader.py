import os
import yaml
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

class UNetCustomDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data_cfg = config['data']
        self.aug_cfg = config['augmentation']
        
        # 获取所有同名文件名
        self.file_names = sorted(os.listdir(os.path.join(self.data_cfg['imgs_dir'], 'ROI')))
        
        # 数据集切分
        self._split_data()

    def _split_data(self):
        random.seed(42) # 固定随机种子确保切分一致
        random.shuffle(self.file_names)
        n = len(self.file_names)
        train_end = int(n * self.data_cfg['splits'][0])
        val_end = train_end + int(n * self.data_cfg['splits'][1])
        
        if self.mode == 'train':
            self.file_names = self.file_names[:train_end]
        elif self.mode == 'val':
            self.file_names = self.file_names[train_end:val_end]
        else:
            self.file_names = self.file_names[val_end:]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        
        # 1. 加载图像 (使用 OpenCV 读取灰度图)
        # 特征集路径
        roi_path = os.path.join(self.data_cfg['imgs_dir'], 'ROI', name)
        img_mask_path = os.path.join(self.data_cfg['imgs_dir'], 'Mask', name)
        # 标签集路径
        lab_mask_path = os.path.join(self.data_cfg['labels_dir'], 'Mask', name)
        phi_w_path = os.path.join(self.data_cfg['labels_dir'], 'Phi_w', name)

        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
        lab_mask = cv2.imread(lab_mask_path, cv2.IMREAD_GRAYSCALE)
        phi_w = cv2.imread(phi_w_path, cv2.IMREAD_GRAYSCALE)

        # 转换为 PIL 以便使用 torchvision 变换
        roi = F.to_pil_image(roi)
        img_mask = F.to_pil_image(img_mask)
        lab_mask = F.to_pil_image(lab_mask)
        phi_w = F.to_pil_image(phi_w)

        # 2. 执行数据增强 (特征与标签同步)
        if self.aug_cfg['enabled'] and self.mode == 'train':
            roi, img_mask, lab_mask, phi_w = self._apply_augmentation(roi, img_mask, lab_mask, phi_w)

        # 3. 特定处理：对 ROI 进行对比度调整
        if self.aug_cfg['enabled'] and self.mode == 'train':
            c_min, c_max = self.aug_cfg['contrast_range']
            contrast_factor = random.uniform(c_min, c_max)
            roi = F.adjust_contrast(roi, contrast_factor)

        # 4. 转换为 Tensor 并进行缩放/插值处理
        # 确保大小为 256x256
        roi = F.resize(roi, [256, 256], interpolation=F.InterpolationMode.BILINEAR)
        
        # Mask 使用最近邻插值并保持二值化
        img_mask = F.resize(img_mask, [256, 256], interpolation=F.InterpolationMode.NEAREST)
        lab_mask = F.resize(lab_mask, [256, 256], interpolation=F.InterpolationMode.NEAREST)

        # phi_w 使用最近邻插值并保持二值化
        phi_w = F.resize(phi_w, [256, 256], interpolation=F.InterpolationMode.NEAREST)

        # 5. 特殊映射逻辑：Phi_w -> sin, cos
        phi_array = np.array(phi_w).astype(np.float32) / 255.0 * 2 * np.pi - np.pi # 映射到 [-pi, pi]
        sin_phi = np.sin(phi_array)
        cos_phi = np.cos(phi_array)

        # 6. 通道合并
        # 输入：ROI (0), Mask (1)
        input_tensor = torch.stack([F.to_tensor(roi)[0], F.to_tensor(img_mask)[0]], dim=0)
        
        # 标签：Mask (0), Sin (1), Cos (2)
        mask_tensor = (F.to_tensor(lab_mask) > 0.5).float() # 确保二值化
        label_tensor = torch.cat([
            mask_tensor, 
            torch.from_numpy(sin_phi).unsqueeze(0), 
            torch.from_numpy(cos_phi).unsqueeze(0)
        ], dim=0)

        return input_tensor, label_tensor

    def _apply_augmentation(self, roi, img_m, lab_m, phi):
        # 1. 随机水平翻转
        if random.random() > 0.5:
            roi, img_m, lab_m, phi = F.hflip(roi), F.hflip(img_m), F.hflip(lab_m), F.hflip(phi)
            
        # 2. 随机垂直翻转
        if random.random() > 0.5:
            roi, img_m, lab_m, phi = F.vflip(roi), F.vflip(img_m), F.vflip(lab_m), F.vflip(phi)
        
        # 3. 随机旋转
        angle = random.uniform(-self.aug_cfg['rotation_range'], self.aug_cfg['rotation_range'])
        # fill=0 对应你要求的补0处理
        roi = F.rotate(roi, angle, fill=0)
        img_m = F.rotate(img_m, angle, fill=0)
        lab_m = F.rotate(lab_m, angle, fill=0)
        phi = F.rotate(phi, angle, fill=0)

        # 4. 随机裁剪
        min_crop_scale = self.aug_cfg['min_crop_scale']
        # 获取随机裁剪的参数 i, j, h, w
        i, j, h, w = T.RandomResizedCrop.get_params(
            roi, 
            scale=(min_crop_scale, 1.0), 
            ratio=(0.5, 2)
        )
        # 应用裁剪
        roi = F.crop(roi, i, j, h, w)
        img_m = F.crop(img_m, i, j, h, w)
        lab_m = F.crop(lab_m, i, j, h, w)
        phi = F.crop(phi, i, j, h, w)

        return roi, img_m, lab_m, phi
def get_dataloaders(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    modes = ['train', 'val', 'test']
    loaders = {}
    
    for mode in modes:
        dataset = UNetCustomDataset(config, mode=mode)
        loaders[mode] = DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=(mode == 'train' and config['augmentation']['shuffle']),
            num_workers=config['data']['num_workers']
        )
    return loaders

# 测试数据加载器
if __name__ == '__main__':

    # 获取加载器
    loaders = get_dataloaders('config.yaml')
    train_loader = loaders['train']
    val_loader = loaders['val']

    for inputs, targets in train_loader:
        # inputs 形状: [B, 2, 256, 256] -> (ROI, Mask)
        # targets 形状: [B, 3, 256, 256] -> (Mask, Sin, Cos)
        print(inputs.shape, targets.shape)
        # 可视化检查
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs[0, 0].imshow(inputs[0, 0].numpy(), cmap='gray')
        axs[0, 0].set_title('ROI')
        axs[0, 1].imshow(inputs[0, 1].numpy(), cmap='gray')
        axs[0, 1].set_title('Input Mask')
        axs[0, 2].imshow(targets[0, 0].numpy(), cmap='gray')
        axs[0, 2].set_title('Label Mask')
        axs[1, 0].imshow(np.arctan2(targets[0, 2].numpy(), targets[0, 1].numpy()), cmap='gray')
        axs[1, 0].set_title('Phi (atan2)')
        axs[1, 1].imshow(targets[0, 1].numpy(), cmap='gray')
        axs[1, 1].set_title('Sin(Phi)')
        axs[1, 2].imshow(targets[0, 2].numpy(), cmap='gray')
        axs[1, 2].set_title('Cos(Phi)')
        
        # 移除坐标轴
        for ax in axs.flat:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

        

        break