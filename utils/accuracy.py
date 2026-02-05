import torch
import torch.nn as nn

def accuracy(pred, target):
    """
    计算预测结果的准确率。
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def mae(pred, target):
    """
    计算预测结果的平均绝对误差（MAE）。
    """
    return torch.abs(pred - target).mean()
