import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=('relu1_1', 'relu2_1', 'relu3_1'), use_gpu=True):
        super().__init__()
        # 加载预训练 VGG19 并只保留特征提取部分
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg.eval()  # 切到 eval 模式
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        # 将我们关心的层名映射到索引
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11,'relu3_2': 13,'relu3_3': 15,'relu3_4': 17,
            'relu4_1': 20,'relu4_2': 22,'relu4_3': 24,'relu4_4': 26,
            'relu5_1': 29
        }
        self.layer_ids = [layer_map[l] for l in layers]
        self.criterion = nn.L1Loss()

        if use_gpu:
            self.vgg_layers = self.vgg_layers.cuda()

    def forward(self, input, target):
        """
        input, target: 都是归一化到 [-1,1] 的图像张量 [B,3,H,W]
        """
        # 把范围从 [-1,1] 转到 [0,1]，再做 ImageNet normalization
        def normalize(x):
            return (x * 0.5 + 0.5 - torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)) \
                   / torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)

        x = normalize(input)
        y = normalize(target)

        loss = 0.0
        xi, yi = x, y
        for idx, layer in enumerate(self.vgg_layers):
            xi = layer(xi)
            yi = layer(yi)
            if idx in self.layer_ids:
                loss += self.criterion(xi, yi)
        return loss
