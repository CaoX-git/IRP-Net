import os
import yaml
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from utils.builder import build_model
import matplotlib.pyplot as plt

def predict(archive_path, roi_path, mask_path = None):
    # 1. 加载配置与设备

    config_path = os.path.join(archive_path, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 2. 加载模型
    model = build_model(config).to(device)

    # 3. 加载权重
    checkpoint_path = os.path.join(archive_path, 'checkpoints', 'best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到权重文件: {checkpoint_path}")
        return

    print(f"🔄 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 4. 应用权重
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 5. 读取输入图像
    # 读取灰度图
    roi_cv = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    # 直方图均衡化
    # roi_cv = cv2.equalizeHist(roi_cv)

    if mask_path is None:
        mask_cv = np.ones_like(roi_cv)*255
    else:
        mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if roi_cv is None or mask_cv is None:
        print("❌ 图像路径无效，请检查！")
        return

    # 预处理：缩放并转换为 Tensor
    roi_p = cv2.resize(roi_cv, (256, 256))
    mask_p = cv2.resize(mask_cv, (256, 256))

    # 通道合并: ROI (0), Mask (1)
    input_tensor = torch.stack([
        F.to_tensor(roi_p)[0], 
        F.to_tensor(mask_p)[0]
    ], dim=0).unsqueeze(0).to(device) # 增加 Batch 维度 [1, 2, 256, 256]

    # 6. 模型推理
    with torch.no_grad():
        output = model(input_tensor) # 输出维度: [1, 3, 256, 256]
        output = output.squeeze(0).cpu() # 移出 Batch 维度并转回 CPU

    # 7. 后处理逻辑
    # 通道 0: Mask (预测)
    pred_mask = torch.sigmoid(output[0]).numpy() 
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255 # 二值化

    # 通道 1, 2: Sin, Cos -> 计算包裹相位
    pred_sin = output[1].numpy()
    pred_cos = output[2].numpy()
    
    # 使用 arctan2(sin, cos) 计算相位，范围为 [-pi, pi]
    pred_phase = np.arctan2(pred_sin, pred_cos)

    # 映射相位 [-pi, pi] 到 [0, 255] 以便显示
    pred_phase_norm = ((pred_phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

    # 叠加掩码
    # 输入 Mask 叠加到 ROI 上
    roi_masked = cv2.bitwise_and(roi_p, roi_p, mask=(mask_p > 127).astype(np.uint8) * 255)

     # 预测 Mask 叠加到 ROI 上
    pred_masked = cv2.bitwise_and(pred_phase_norm, pred_phase_norm, mask=(pred_mask > 127).astype(np.uint8) * 255)

    # 8. 显示与保存
    # 拼接图像进行对比展示: 输入 ROI | 输入 Mask | 预测 Mask | 预测相位
    combined_img = np.hstack([roi_masked, pred_masked])

    # 保存结果
    save_name = os.path.basename(archive_path)
    save_path = os.path.join(config.get('predict', {}).get('save_dir', 'results'), f'{save_name}_prediction_result.png')
    cv2.imwrite(save_path, combined_img)
    print(f"✅ Result saved to: {save_path}")

    # 显示图像
    plt.figure(figsize=(8, 4))
    # BGR转RGB，适配matplotlib显示格式
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title("Inference Result (input | output)")  
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 弹出窗口显示图像

if __name__ == "__main__":
    archive_path = "archive/UNet_260207161748"
    roi_path = "data_template/dataset_experiment/img1.png"
    mask_path = None

    # roi_path = "data_template/dataset_template/images/ROI/1_4.658.png"
    # mask_path = "data_template/dataset_template/images/Mask/1_4.658.png"

    predict(archive_path, roi_path, mask_path)    

