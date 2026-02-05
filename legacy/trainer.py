# trainer.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from src.loss import get_loss_function
from utils.accuracy import accuracy, mae
from utils.data_loader import create_dataloaders_from_yaml
from model import UNet
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimizer(model, optimizer_name: str, learning_rate: float):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_config: dict):
    if scheduler_config["type"].lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            imgs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()  # 清除上一批次的梯度

            # 模型输出
            predict = model(imgs)

            # 计算损失
            loss = loss_fn(predict, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 计算准确率（仅适用于二分类任务）
            acc = mae(predict, labels)
            correct += acc.item()
            total += 1

            tepoch.set_postfix(loss=loss.item(), acc=acc)
            del batch
        torch.cuda.empty_cache()


    return epoch_loss / len(dataloader), correct / total


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(dataloader, unit="batch"):
            imgs, labels = batch["image"].to(device), batch["label"].to(device)

            # 模型输出
            predict = model(imgs)

            # 计算损失
            loss = loss_fn(predict, labels)
            val_loss += loss.item()

            # 计算准确率
            acc = mae(predict, labels)
            correct += acc.item()
            total += 1

    return val_loss / len(dataloader), correct / total


def train(config_path: str):
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    training_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    

    # 数据加载
    loaders = create_dataloaders_from_yaml(config_path, batch_size=training_cfg["batch_size"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 模型初始化
    model = UNet(in_ch=data_cfg["channels"], num_classes=model_cfg["num_classes"], bilinear=model_cfg["bilinear"], base_c=model_cfg["base_channels"]).to(device)

    # 获取优化器
    optimizer = get_optimizer(
        model, training_cfg["optimizer"], training_cfg["learning_rate"]
    )

    # 获取调度器
    scheduler = get_scheduler(optimizer, training_cfg["scheduler"])

    # 获取损失函数
    loss_fn = get_loss_function(training_cfg["loss"]["type"])

    # 训练
    best_val_loss = float("inf")
    for epoch in range(training_cfg["num_epochs"]):
        print(f"Epoch {epoch+1}/{training_cfg['num_epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if epoch == 0:
            # 在指定路径下加入时间戳目录
            model_cfg["weights_dir"] = os.path.join(model_cfg["weights_dir"], time.strftime("%Y%m%d_%H%M%S", time.localtime()))
            if not os.path.exists(model_cfg["weights_dir"]):
                os.makedirs(model_cfg["weights_dir"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(model_cfg["weights_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print("Saved best model!")

    print("Training complete!")


if __name__ == "__main__":
    # 默认配置路径为 config.yaml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(config_path="config.yaml")
