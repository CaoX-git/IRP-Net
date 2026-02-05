
import torch
import torch.nn as nn
import yaml
from utils import loss as custom_losses
from utils import evaluator as custom_evaluators
import torch.optim as optim
import models

# 模型构建器
def build_model(config):
    """
    根据配置动态构建模型
    """

    model_cfg = config.get('model', {})
    model_name = model_cfg.get('type')
    model_params = model_cfg.get('params', {})

    # 动态获取模型类
    try:
        # 从 models 包中通过字符串名称获取类对象
        model_class = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"在 models/ 中找不到模型类: {model_name}。请检查 __init__.py 是否已导入该类。")

    return model_class(**model_params)

# 损失函数构建器
def build_criterion(config):
    """
    根据配置动态构建损失函数
    """
    training_cfg = config.get('training', {})
    loss_name = training_cfg.get('loss_type', {})
    loss_params = training_cfg.get('loss_params', {})

    # 逻辑：先从当前文件找，找不到再去 torch.nn 找
    if hasattr(custom_losses, loss_name):
        loss_class = getattr(custom_losses, loss_name)
    elif hasattr(nn, loss_name):
        loss_class = getattr(nn, loss_name)
    else:
        raise ValueError(f"未找到指定的损失函数: {loss_name}")

    # 如果没有指定参数，直接返回损失函数实例
    if loss_params == None:
        return loss_class()
    else:
        return loss_class(**loss_params)

# 优化器构建器
def build_optimizer(config, model):
    """
    根据配置动态构建优化器
    """
    training_cfg = config.get('training', {})
    optim_cfg = training_cfg.get('optimizer', {})
    name = optim_cfg['type']
    base_params = {'lr': optim_cfg['lr'], 'weight_decay': optim_cfg.get('weight_decay', 0)}
    extra_params = optim_cfg.get('extra_params', {})
    
    opt_class = getattr(optim, name)
    return opt_class(model.parameters(), **base_params, **extra_params)

def build_evaluator(config):
    """
    根据配置动态构建评估器
    """
    training_cfg = config.get('training', {})
    evaluator_name = training_cfg.get('evaluator_type', {})
    extra_params = training_cfg.get('evaluator_params', {})
    if evaluator_name == None:
        return None
    evaluator_class = getattr(custom_evaluators, evaluator_name)

    if extra_params == None:
        return evaluator_class()
    else:
        return evaluator_class(**extra_params)

# 测试构建器
if __name__ == "__main__":
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)


    # region 测试模型选择器与优化器构建
    print("-"*30)
    model = build_model(config)
    print(f"当前使用的模型是: {type(model).__name__}")
    optimizer = build_optimizer(config, model)
    print(f"当前使用的优化器是: {type(optimizer).__name__}")
    print("-"*30)
    # endregion


    # region 测试自定义损失函数与评估器构建
    criterion = build_criterion(config)
    evaluator = build_evaluator(config)
    print(f"当前使用的损失函数是: {type(criterion).__name__}")
    print(f"当前使用的评估器是: {type(evaluator).__name__}")
    print("-"*30)

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
    print("-"*30)
    print(f"Computed Loss: {loss.item():.6f}")
    
    # 全量评估
    results = evaluator(mock_pred, mock_target)
    # 打印指标
    print("-"*30)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 反向传播测试
    loss.backward()
    print("-"*30)
    print("Backward pass successful.")
    print("-"*30)
    # endregion


