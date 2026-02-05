import torch
import torch.nn as nn
import torch.nn.functional as F

# region CBAM_ResUNet

# CBAM Block
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        hidden = max(channels // reduction_ratio, 1)

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

        # cache for visualization
        self.last_channel_att = None   # [B, C, 1, 1]
        self.last_spatial_att = None   # [B, 1, H, W]

    def forward(self, x):
        b, c, _, _ = x.shape

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))

        self.last_channel_att = channel_att.detach()
        self.last_spatial_att = spatial_att.detach()

        return x * spatial_att

# ConvBNReLU Block
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

# Residual Block
class ResBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(ResBlock, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        # 主路径：两层卷积
        self.conv_path = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        # 捷径路径：处理输入输出通道不一致的情况
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        # 最后使用激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_path(x)
        out += self.shortcut(x)
        return self.relu(out)

# Encoder Block
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        self.double_conv = ResBlock(in_ch, out_ch, mid_ch)
        self.CBAM = CBAM(out_ch)
        self.down = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.double_conv(x)
        x = self.CBAM(x)
        x = self.down(x)
        return x

# Decoder Block
class Decoder(nn.Module):
    def __init__(self, in_ch, concat_ch, out_ch, mid_ch=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ConcatConv = ResBlock(in_ch + concat_ch, out_ch, mid_ch)
        self.CBAM = CBAM(out_ch)
    
    def forward(self, x, concat):
        x = self.up(x)
        x = torch.cat([x, concat], dim=1)
        x = self.ConcatConv(x)
        x = self.CBAM(x)
        return x

# CBAM_ResUNet
class CBAM_ResUNet(nn.Module):
    """
    输入 [B,in_ch,H,W] 输出 [B,out_ch,H,W]
    """
    def __init__(self, in_ch=1, out_ch=2, bilinear=True, base_c=64):
        super().__init__()
        self.input = ResBlock(in_ch, base_c)
        self.enc1 = Encoder(base_c, base_c * 2)
        self.enc2 = Encoder(base_c * 2, base_c * 4)
        self.enc3 = Encoder(base_c * 4, base_c * 8)
        self.enc4 = Encoder(base_c * 8, base_c * 16)
        self.bridge = ResBlock(base_c * 16, base_c * 16)
        self.up1 = Decoder(base_c * 16, base_c * 8, base_c * 8)
        self.up2 = Decoder(base_c * 8, base_c * 4, base_c * 4)
        self.up3 = Decoder(base_c * 4, base_c * 2, base_c * 2)
        self.up4 = Decoder(base_c * 2, base_c, base_c)
        self.out = nn.Conv2d(base_c, out_ch, kernel_size=1)

    def forward(self, ori_in):
        # 干涉区域与掩码区域通道分离
        roi = ori_in[:, 0:1, :, :] 
        mask = ori_in[:, 1:2, :, :]

        # 干涉区域输入编码器前向传播
        x = roi
        x1 = self.input(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.bridge(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)

        # 掩码区域与输出通道拼接
        final_out = torch.cat([mask, out], dim=1)
        return final_out

# endregion

# 测试模型
if __name__ == '__main__':
    model = CBAM_ResUNet(in_ch=1, out_ch=2, bilinear=True, base_c=64).to('cuda')
    print("-" * 30)
    print(model)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 30)
    print(f"模型参数数量: {total_params / 1024 / 1024:.2f} M")

    # 测试输入输出尺寸
    test_input = torch.randn(8, 2, 256, 256).to('cuda')

    # 统计初始显存
    start_mem = torch.cuda.memory_allocated() / 1024**2
    print("-" * 30)
    print(f"初始显存占用: {start_mem:.2f} MB")

    # 运行模型一次
    test_output = model(test_input)
    print(test_output.shape)

    # 统计峰值显存
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    cost_mem = peak_mem - start_mem

    print("-" * 30)
    print(f"训练峰值显存: {peak_mem:.2f} MB")
    print(f"训练消耗显存: {cost_mem:.2f} MB")
    print("-" * 30)