import torch
import torch.nn as nn

# 掩码加权均方误差损失函数
class MaskWeightedMSELoss(nn.Module):
    """
    掩码加权损失函数
    用于在 ROI 区域内计算预测值与真实值之间的归一化残差平方和
    """
    def __init__(self, epsilon=1e-8):
        super(MaskWeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 1. 提取掩码和 sincos 通道
        target_mask = target[:, 0:1 :, :]
        target_sincos = target[:, 1:3 :, :]
        pred_sincos = pred[:, 1:3 :, :]

        # 2. 计算残差并应用 Hadamard 积 (⊙)
        # 将 ROI 区域外的误差强制置零
        diff = target_mask * (target_sincos - pred_sincos)
        
        # 3. 计算残差平方和 (||·||2)
        # 对应公式分子：Px,y |M_ROI * (Y_GT - Y_pred)|^2
        squared_errors = torch.pow(diff, 2)
        sum_squared_errors = torch.sum(squared_errors, dim=(1, 2, 3))
        
        # 4. 计算 ROI 区域内的有效像素总数
        # 对应公式分母：Px,y M_ROI
        roi_pixel_count = torch.sum(target_mask, dim=(1, 2, 3))
        
        # 5. 归一化并添加 epsilon 防止除零(这里实际计算量sin和cos两个通道里的有效像素，所以需要除以2)
        # 对应公式括号内的部分
        loss_per_sample = sum_squared_errors / (2 * roi_pixel_count + self.epsilon)
        
        # 6. 对整个 Batch 取平均 (1/B)
        total_loss = torch.mean(loss_per_sample)
        
        return total_loss

# 测试损失函数构建
if __name__ == "__main__":

    criterion = MaskWeightedMSELoss()

    # 模拟数据
    batch_size = 4
    channels = 2
    height, width = 256, 256
    # 随机生成预测值、真实值和掩码
    mock_pred_sincos = torch.randn(batch_size, channels, height, width, requires_grad=True)
    mock_target_sincos = torch.randn(batch_size, channels, height, width)
    # 掩码是 0 和 1
    mock_mask = (torch.randn(batch_size, 1, height, width) > 0).float()

    mock_pred = torch.cat([mock_mask, mock_pred_sincos], dim=1)
    mock_target = torch.cat([mock_mask, mock_target_sincos], dim=1)
    
    loss = criterion(mock_pred, mock_target)
    print(f"Computed Loss: {loss.item():.6f}")
    
    # 反向传播测试
    loss.backward()
    print("Backward pass successful.")

