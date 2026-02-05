import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

class PhaseEvaluator:
    def __init__(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), mask_threshold: float = 0.5):
        """
        相位评估工具类
        :param device: 计算设备 (cuda/cpu)
        :param mask_threshold: 掩码有效区域阈值
        """
        self.mask_thresh = mask_threshold
        self.device = device
        # 初始化图像质量评估指标
        self.ssim = SSIM(data_range=1.0).to(device)
        self.psnr = PSNR(data_range=1.0).to(device)

    def get_valid_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """获取有效区域掩码 (bool张量)"""
        return mask_tensor > self.mask_thresh

    def reconstruct_phase(self, sin_map: torch.Tensor, cos_map: torch.Tensor) -> torch.Tensor:
        """从sin/cos还原包裹相位 φ ∈ [-π, π]"""
        phi = torch.atan2(sin_map, cos_map)
        return phi

    # ==================== 基础连续值指标（sin/cos通道） ====================
    def compute_sin_cos_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """计算sin、cos通道的MSE/RMSE/MAE（仅有效区域）"""
        # 拆分通道: [B, 3, H, W] -> mask, sin, cos
        mask_pred, sin_pred, cos_pred = pred.chunk(3, dim=1)
        mask_gt, sin_gt, cos_gt = target.chunk(3, dim=1)

        # 有效掩码
        valid_mask = self.get_valid_mask(mask_gt)
        if valid_mask.sum() == 0:
            return {"mse": 0.0, "rmse": 0.0, "mae": 0.0}

        # 提取有效区域数值
        sin_pred_valid = sin_pred[valid_mask]
        sin_gt_valid = sin_gt[valid_mask]
        cos_pred_valid = cos_pred[valid_mask]
        cos_gt_valid = cos_gt[valid_mask]

        # 合并sin+cos误差
        pred_all = torch.cat([sin_pred_valid, cos_pred_valid])
        gt_all = torch.cat([sin_gt_valid, cos_gt_valid])

        mse = F.mse_loss(pred_all, gt_all)
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(pred_all, gt_all)

        return {
            "sin_cos_mse": mse.item(),
            "sin_cos_rmse": rmse.item(),
            "sin_cos_mae": mae.item()
        }

    # ==================== 相位专用周期指标 ====================
    def compute_phase_wrapped_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """计算包裹相位的最小圆周误差指标"""
        mask_pred, sin_pred, cos_pred = pred.chunk(3, dim=1)
        mask_gt, sin_gt, cos_gt = target.chunk(3, dim=1)

        valid_mask = self.get_valid_mask(mask_gt)
        if valid_mask.sum() == 0:
            return {"wrapped_rmse": 0.0, "angular_mae": 0.0}

        # 还原相位
        phi_pred = self.reconstruct_phase(sin_pred, cos_pred)
        phi_gt = self.reconstruct_phase(sin_gt, cos_gt)

        # 计算最小圆周误差
        delta = phi_pred - phi_gt
        circular_error = torch.atan2(torch.sin(delta), torch.cos(delta))

        # 仅有效区域
        circular_error_valid = circular_error[valid_mask]
        wrapped_rmse = torch.sqrt(torch.mean(circular_error_valid ** 2))
        angular_mae = torch.mean(torch.abs(circular_error_valid))

        return {
            "wrapped_rmse": wrapped_rmse.item(),
            "angular_mae": angular_mae.item()
        }

    # ==================== 图像结构指标 ====================
    def compute_image_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """计算SSIM/PSNR，使用还原后的相位单通道图"""
        _, sin_pred, cos_pred = pred.chunk(3, dim=1)
        _, sin_gt, cos_gt = target.chunk(3, dim=1)

        # 还原相位并归一化到 [0,1] 适配SSIM/PSNR
        phi_pred = self.reconstruct_phase(sin_pred, cos_pred)
        phi_gt = self.reconstruct_phase(sin_gt, cos_gt)
        phi_pred_norm = (phi_pred + torch.pi) / (2 * torch.pi)
        phi_gt_norm = (phi_gt + torch.pi) / (2 * torch.pi)

        ssim_val = self.ssim(phi_pred_norm, phi_gt_norm).item()
        psnr_val = self.psnr(phi_pred_norm, phi_gt_norm).item()

        return {"ssim": ssim_val, "psnr": psnr_val}

    # ==================== 总接口 ====================
    def evaluate_all(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """计算所有评估指标，统一输出"""
        metrics = {}
        metrics.update(self.compute_sin_cos_metrics(pred, target))
        metrics.update(self.compute_phase_wrapped_metrics(pred, target))
        metrics.update(self.compute_image_metrics(pred, target))
        return metrics
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """计算所有评估指标，统一输出"""
        return self.evaluate_all(pred, target)

class NoEvaluator(torch.nn.Module):
    def __init__(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NoEvaluator, self).__init__()
        self.device = device

    def forward(self, pred, target):
        return 0

if __name__ == "__main__":
    # 测试评估器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = PhaseEvaluator(device=device)

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

    # 全量评估
    results = evaluator.evaluate_all(mock_pred, mock_target)
    # 打印指标
    for k, v in results.items():
        print(f"{k}: {v:.4f}")