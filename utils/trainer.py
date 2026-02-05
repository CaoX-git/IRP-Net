import torch
import os
import csv
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, 
                 evaluator=None, scheduler=None, save_dir="results", patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator 
        self.scheduler = scheduler # 学习率调度器
        self.save_dir = save_dir
        
        # 路径管理
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        self.log_path = os.path.join(save_dir, "train_log.csv")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # 训练状态变量
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        
        # 早停相关
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def _init_log_file(self, resume=False):
        """初始化或检查日志文件"""
        if not resume or not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_metric'])

    def log_to_csv(self, epoch, lr, train_loss, val_loss, val_metric):
        """追加数据"""
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{lr:.8f}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_metric:.6f}"])

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}] Train")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                if self.evaluator:
                    all_preds.append(outputs.cpu())
                    all_targets.append(targets.cpu())
        
        avg_loss = val_loss / len(self.val_loader)
        metric = self.evaluator(torch.cat(all_preds), torch.cat(all_targets)) if self.evaluator else 0
        return avg_loss, metric

    def fit(self, epochs, resume_path=None):
        """主训练入口，支持续训"""
        if resume_path:
            self.load_checkpoint(resume_path)
            self._init_log_file(resume=True)
        else:
            self._init_log_file(resume=False)

        for epoch in range(self.start_epoch, epochs + 1):
            lr = self.optimizer.param_groups[0]['lr']
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metric = self.validate(epoch)

            # 更新学习率 (如果有 scheduler)
            if self.scheduler:
                self.scheduler.step(val_loss) # 假设使用 ReduceLROnPlateau

            # 记录数据
            self.log_to_csv(epoch, lr, train_loss, val_loss, val_metric)
            
            # 保存权重
            self.save_checkpoint(epoch, val_loss)

            # 早停逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                print(f"🌟 Improved! Saved Best Model.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break

    def save_checkpoint(self, epoch, val_loss):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss
        }
        torch.save(state, os.path.join(self.checkpoint_dir, 'last.pth'))
        if val_loss <= self.best_val_loss:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best.pth'))

    def load_checkpoint(self, path):
        """续训核心：恢复所有状态"""
        print(f"🔄 Resuming from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"✅ Resumed from Epoch {checkpoint['epoch']}")
