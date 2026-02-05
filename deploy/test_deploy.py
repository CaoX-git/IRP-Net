import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import onnxruntime as ort
import matplotlib.pyplot as plt

def preprocess(roi_path, mask_path=None, size=(256, 256)):
    """ 保持与 predict.py 完全一致的预处理 """
    roi_cv = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if mask_path is None:
        mask_cv = np.ones_like(roi_cv) * 255
    else:
        mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if roi_cv is None:
        raise ValueError(f"无法读取图像: {roi_path}")

    # 缩放
    roi_p = cv2.resize(roi_cv, size)
    mask_p = cv2.resize(mask_cv, size)

    # 转换为 Tensor 并堆叠 [1, 2, 256, 256]
    # 通道 0: ROI, 通道 1: Mask
    input_tensor = torch.stack([
        F.to_tensor(roi_p)[0], 
        F.to_tensor(mask_p)[0]
    ], dim=0).unsqueeze(0) 
    
    return input_tensor, roi_p

def test_torchscript(model_path, input_tensor):
    print(f"🔍 正在测试 TorchScript: {model_path}")
    # 加载模型 (不需要 models/*.py)
    model = torch.jit.load(model_path)
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy()

def test_onnx(model_path, input_tensor):
    print(f"🔍 正在测试 ONNX: {model_path}")
    # 创建推理会话
    providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 准备输入字典
    input_name = session.get_inputs()[0].name
    input_data = input_tensor.numpy()
    
    # 推理
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]

def main():
    # --- 配置 ---
    # 图片路径
    ROI_PATH = "data_template/dataset_experiment/img1.png"
    MASK_PATH = None
    
    # 指向导出的文件（根据 export.py 生成的文件名修改）
    PT_MODEL = "exports/pt/CBAM_ResUNet_nobridge_best.pt" 
    ONNX_MODEL = "exports/onnx/CBAM_ResUNet_nobridge_best.onnx"
    
    # 1. 预处理数据
    input_tensor, roi_p = preprocess(ROI_PATH, MASK_PATH)
    print(f"✅ 输入数据准备完成: {input_tensor.shape}")

    results = {}

    # 2. 测试 TorchScript
    if os.path.exists(PT_MODEL):
        pt_out = test_torchscript(PT_MODEL, input_tensor)
        results['TorchScript'] = pt_out
        print(f"✅ TorchScript 输出形状: {pt_out.shape}")
    else:
        print(f"⚠️ 未找到 PT 模型: {PT_MODEL}")

    # 3. 测试 ONNX
    if os.path.exists(ONNX_MODEL):
        onnx_out = test_onnx(ONNX_MODEL, input_tensor)
        results['ONNX'] = onnx_out
        print(f"✅ ONNX 输出形状: {onnx_out.shape}")
    else:
        print(f"⚠️ 未找到 ONNX 模型: {ONNX_MODEL}")

    # 4. 简单可视化对比（如果两个都有）
    if 'TorchScript' in results and 'ONNX' in results:
        diff = np.abs(results['TorchScript'] - results['ONNX']).mean()
        print(f"📊 PT 与 ONNX 平均误差 (Mean Absolute Error): {diff:.2e}")
        
        # 显示两者的第二个输出通道
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input ROI")
        plt.imshow(roi_p, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("PT Output (Ch 2)")
        plt.imshow(results['TorchScript'][0, 1], cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("ONNX Output (Ch 2)")
        plt.imshow(results['ONNX'][0, 1], cmap='gray')
        
        plt.show()

if __name__ == "__main__":
    main()