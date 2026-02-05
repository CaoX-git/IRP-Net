# -*- coding: utf-8 -*-
"""
Visualize all CBAM blocks in CBAM_UNet:
- spatial attention heatmap
- spatial attention overlay on input image (auto-resize)
- channel attention bar chart (auto Top-K if too many channels)
- save full attention arrays as .npy for further analysis

Usage example:
python visualize_cbam_unet.py \
  --model-ckpt pth/CBAM_UNet.pth \
  --image path/to/test.png \
  --output-dir ./cbam_vis \
  --device cuda \
  --img-size 256 \
  --in-ch 1 \
  --num-classes 2 \
  --base-c 64 \
  --bilinear 1
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) 改成你的模型文件名
#    例如你把模型定义保存为: CBAM_UNet_model.py
from model import CBAM_UNet, CBAM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", type=str, default=r'weights\20251219_003439\best_model.pth')
    p.add_argument("--image", type=str, default=r"C:\Users\41323\Desktop\Tests\Python\DataGen\data\251212_1\IMG_NOISE\1_3.939.png")
    p.add_argument("--output-dir", type=str, default="cbam_vis")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img-size", type=int, default=256)

    # ---- 和你训练时保持一致 ----
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--base-c", type=int, default=64)
    p.add_argument("--bilinear", type=int, default=1, help="1=True, 0=False")

    # 可视化控制
    p.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    p.add_argument("--topk-ch", type=int, default=128, help="plot top-k channels if channels too many")
    p.add_argument("--max-ch-plot", type=int, default=256, help="if C>max, only plot topk")
    return p.parse_args()


def _strip_module_prefix(state_dict):
    # 兼容 DataParallel / DDP 保存的 'module.xxx'
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_to_model(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容多种保存格式
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    ckpt = _strip_module_prefix(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print("[Warn] Missing keys (show first 20):", missing[:20])
    if unexpected:
        print("[Warn] Unexpected keys (show first 20):", unexpected[:20])


def load_model(args):
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model = CBAM_UNet(
        in_ch=args.in_ch,
        num_classes=args.num_classes,
        bilinear=bool(args.bilinear),
        base_c=args.base_c
    ).to(device)
    model.eval()

    load_checkpoint_to_model(model, args.model_ckpt, device)
    print(f"Loaded model ckpt: {args.model_ckpt}")
    return model, device


def load_image_as_tensor(image_path: str, img_size: int, in_ch: int):
    """
    返回:
      x: [1, C, H, W] torch tensor
      orig_np: [H, W] 或 [H, W, 3] in [0,1] for visualization
    """
    if in_ch == 1:
        img = Image.open(image_path).convert("L")
        tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])  # [1,H,W]
        x = tfm(img).unsqueeze(0)  # [1,1,H,W]
        orig = np.array(img.resize((img_size, img_size)), dtype=np.float32) / 255.0  # [H,W]
        return x, orig

    elif in_ch == 3:
        img = Image.open(image_path).convert("RGB")
        tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])  # [3,H,W]
        x = tfm(img).unsqueeze(0)  # [1,3,H,W]
        orig = np.array(img.resize((img_size, img_size)), dtype=np.float32) / 255.0  # [H,W,3]
        return x, orig

    else:
        # 其他通道数：先读灰度，然后复制到 in_ch
        img = Image.open(image_path).convert("L")
        tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])  # [1,H,W]
        x1 = tfm(img)  # [1,H,W]
        x = x1.repeat(in_ch, 1, 1).unsqueeze(0)  # [1,C,H,W]
        orig = np.array(img.resize((img_size, img_size)), dtype=np.float32) / 255.0
        return x, orig


def normalize_01(arr: np.ndarray):
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)


def resize_orig_for_overlay(orig, target_hw):
    H, W = target_hw
    if orig.ndim == 2:
        img = Image.fromarray((orig * 255).astype(np.uint8))
        img = img.resize((W, H), resample=Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    else:
        img = Image.fromarray((orig * 255).astype(np.uint8))
        img = img.resize((W, H), resample=Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


def save_spatial_heatmap(sp_att, save_path, title):
    sp = normalize_01(sp_att)
    plt.figure()
    plt.imshow(sp, cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_spatial_overlay(sp_att, orig, save_path, title, alpha=0.45):
    sp = normalize_01(sp_att)
    H, W = sp.shape
    orig_rs = resize_orig_for_overlay(orig, (H, W))

    plt.figure()
    if orig_rs.ndim == 2:
        plt.imshow(orig_rs, cmap="gray")
    else:
        plt.imshow(orig_rs)
    plt.imshow(sp, cmap="jet", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_channel_bar(ch_att, save_path, title, max_ch_plot=256, topk=128):
    """
    ch_att: [C] numpy
    若 C 很大，只画 topk（按权重从大到小），并在标题提示。
    """
    C = ch_att.shape[0]
    if C <= max_ch_plot:
        x = np.arange(C)
        y = ch_att
        plt.figure(figsize=(max(8, C / 32), 3))
        plt.bar(x, y)
        plt.title(title)
        plt.xlabel("Channel index")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        return

    # Top-K
    k = min(topk, C)
    idx = np.argsort(-ch_att)[:k]
    y = ch_att[idx]
    x = np.arange(k)

    plt.figure(figsize=(10, 3))
    plt.bar(x, y)
    plt.title(f"{title} (Top-{k} of C={C})")
    plt.xlabel("Rank (sorted by weight)")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def visualize_all_cbam(model: nn.Module, x: torch.Tensor, orig, out_dir: str,
                       alpha=0.45, max_ch_plot=256, topk=128):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        _ = model(x)

    cbams = [(name, m) for name, m in model.named_modules() if isinstance(m, CBAM)]
    print(f"Found CBAM blocks: {len(cbams)}")

    if len(cbams) == 0:
        print("No CBAM found. Check import / model definition.")
        return

    for i, (name, m) in enumerate(cbams):
        ch_att = m.last_channel_att
        sp_att = m.last_spatial_att

        if ch_att is None or sp_att is None:
            print(f"[Skip] CBAM #{i} ({name}) has empty attention (not used in forward?).")
            continue

        # batch=0
        ch = ch_att[0].view(-1).detach().cpu().numpy()          # [C]
        sp = sp_att[0, 0].detach().cpu().numpy()                # [H,W]

        # 保存原始数组，方便你后处理/做统计
        safe_name = name.replace(".", "_")
        np.save(os.path.join(out_dir, f"{i:02d}_{safe_name}_channel.npy"), ch)
        np.save(os.path.join(out_dir, f"{i:02d}_{safe_name}_spatial.npy"), sp)

        # 1) spatial heatmap
        save_spatial_heatmap(
            sp,
            os.path.join(out_dir, f"{i:02d}_{safe_name}_spatial.png"),
            title=f"CBAM #{i} | {name} | Spatial"
        )

        # 2) spatial overlay
        save_spatial_overlay(
            sp,
            orig,
            os.path.join(out_dir, f"{i:02d}_{safe_name}_overlay.png"),
            title=f"CBAM #{i} | {name} | Overlay",
            alpha=alpha
        )

        # 3) channel bar
        save_channel_bar(
            ch,
            os.path.join(out_dir, f"{i:02d}_{safe_name}_channel.png"),
            title=f"CBAM #{i} | {name} | Channel",
            max_ch_plot=max_ch_plot,
            topk=topk
        )

        print(f"[OK] Saved CBAM #{i}: {name}")


def main():
    args = parse_args()
    model, device = load_model(args)
    x, orig = load_image_as_tensor(args.image, args.img_size, args.in_ch)
    visualize_all_cbam(
        model, x, orig,
        out_dir=args.output_dir,
        alpha=args.alpha,
        max_ch_plot=args.max_ch_plot,
        topk=args.topk_ch
    )
    print("Done.")


if __name__ == "__main__":
    main()
