import os
import yaml
import argparse
import torch

# 导入自定义模块
from utils.builder import build_model, build_criterion, build_optimizer, build_evaluator
from utils.data_loader import get_dataloaders
from utils.trainer import Trainer

def main(args):
    # 1. 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 3. 构建模型
    model = build_model(config).to(device)

    # 4. 构建数据加载器
    # 传入 config_path 方便 get_dataloaders 内部读取路径参数
    loaders = get_dataloaders(args.config)
    train_loader = loaders['train']
    val_loader = loaders['val']

    # 5. 构建损失函数、优化器和评价指标
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    evaluator = build_evaluator(config)

    # 6. (可选) 构建学习率调度器 Scheduler
    # 使用 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', 
        factor=config.get('scheduler', {}).get('factor', 0.5), 
        patience=config.get('scheduler', {}).get('patience', 5),
        min_lr=config.get('scheduler', {}).get('min_lr', 1e-6),
        cooldown=config.get('scheduler', {}).get('cooldown', 2),
    )

    # 7. 实例化 Trainer
    # 这里的参数对应你要求的 __init__ 签名
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        scheduler=scheduler,
        save_dir=config.get('training', {}).get('save_dir', 'results'),
        patience=config.get('training', {}).get('patience', 10)
    )

    # 8. 断点续训逻辑 (Resume)
    # 检查是否有上次中断的权重文件
    resume_path = os.path.join(config.get('training', {}).get('save_dir', 'results'), 'checkpoints', 'last.pth')
    
    if args.resume and os.path.exists(resume_path):
        print(f"🔄 Found checkpoint at {resume_path}, resuming...")
        trainer.fit(epochs=config['training']['epochs'], resume_path=resume_path)
    else:
        print("🚀 Starting a fresh training...")
        trainer.fit(epochs=config['training']['epochs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    
    # 指定配置文件路径
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    
    # 是否开启续训模式的开关
    parser.add_argument('--resume', action='store_true', default=True,
                        help='resume from last checkpoint if exists')
    
    args = parser.parse_args()
    
    main(args)
