import os
import torch
import yaml
import glob
from pathlib import Path

# 导入项目内的构建工具
from utils.builder import build_model

def export_model(checkpoint_path, save_root, config, device="cpu"):
    """
    导出单个 pth 文件为 pt 和 onnx
    """
    # 1. 获取模型名称和文件名 (例如: CBAM_ResUNet / best)
    model_type = config['model']['type']
    pth_name = Path(checkpoint_path).stem
    
    # 2. 准备输出路径
    pt_dir = os.path.join(save_root, "pt")
    onnx_dir = os.path.join(save_root, "onnx")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    # 定义最终输出文件名：模型名_权重名.格式
    base_filename = f"{model_type}_{pth_name}"
    pt_path = os.path.join(pt_dir, f"{base_filename}.pt")
    onnx_path = os.path.join(onnx_dir, f"{base_filename}.onnx")

    print(f"📦 正在处理: {checkpoint_path}")

    # 3. 构建模型结构并加载权重
    try:
        model = build_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 兼容处理：有些 checkpoint 包装在 'state_dict' 键下
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 准备 Dummy Input (根据 predict.py, 输入为 [1, 2, 256, 256])
    # 通道0: ROI, 通道1: Mask
    dummy_input = torch.randn(1, 2, 256, 256).to(device)

    # 5. 导出 TorchScript (.pt) - 携带模型结构
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(pt_path)
        print(f"✅ TorchScript 导出成功: {pt_path}")
    except Exception as e:
        print(f"❌ TorchScript 导出失败: {e}")

    # 6. 导出 ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX 导出成功: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")

def run_batch_export(target_dir, export_root, config_path):
    """
    遍历文件夹下所有 pth 并导出
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 寻找所有的 .pth 文件
    # 模式匹配：target_dir 下级所有子目录中的 checkpoints/*.pth
    pth_files = glob.glob(os.path.join(target_dir, "**", "checkpoints", "*.pth"), recursive=True)
    
    if not pth_files:
        # 如果层级不对，尝试直接在 target_dir 下找
        pth_files = glob.glob(os.path.join(target_dir, "*.pth"))

    if not pth_files:
        print(f"❓ 在 {target_dir} 中未找到任何 .pth 文件")
        return

    print(f"🚀 找到 {len(pth_files)} 个权重文件，准备开始导出...")

    for pth_path in pth_files:
        # 动态修改 config 中的模型类型（可选）
        # 如果你的文件夹名就是模型名，可以取消下面几行的注释：
        # folder_name = Path(pth_path).parents[1].name  # 假设结构是 archive/ModelName/checkpoints/xx.pth
        # config['model']['type'] = folder_name
        
        export_model(pth_path, export_root, config)

if __name__ == "__main__":
    # --- 配置区域 ---
    # 想要扫描的权重根目录
    SOURCE_DIRECTORY = "archive/CBAM_ResUNet_nobridge_260205144124/checkpoints" 
    # 导出结果根目录
    SAVE_DIRECTORY = "exports"
    # 配置文件路径（用于提供模型参数）
    CONFIG_FILE = os.path.join(Path(SOURCE_DIRECTORY).parent, "config.yaml")
    # ----------------

    run_batch_export(SOURCE_DIRECTORY, SAVE_DIRECTORY, CONFIG_FILE)