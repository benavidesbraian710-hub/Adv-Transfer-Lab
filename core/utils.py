import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


# 1. 修正：标准化包装层（去掉了对 self.mean 的原地赋值，防止计算图报错）
class NormalizeWrapper(nn.Module):
    def __init__(self, model):
        super(NormalizeWrapper, self).__init__()
        self.model = model
        # register_buffer 是标准做法，它会随 model.to(device) 自动移动
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # 修正：直接进行运算，不使用 self.mean = ... 这种赋值操作
        # 这样可以保证梯度流正常，且不报 device 错误
        out = (x - self.mean) / self.std
        return self.model(out)


def get_models(device):
    """加载模型并穿上“标准化的外衣”"""
    # 加载原始模型
    raw_src = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    raw_tgt = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # 包装模型并移动到设备，随后设为 eval 模式
    # 先包装再 .to(device)，register_buffer 就会生效
    src = NormalizeWrapper(raw_src).to(device).eval()
    tgt = NormalizeWrapper(raw_tgt).to(device).eval()

    # 冻结参数
    for p in src.parameters(): p.requires_grad = False
    for p in tgt.parameters(): p.requires_grad = False

    return src, tgt


def run_attack_batch(loader, attacker, src_model, tgt_model, device, desc="Attacking"):
    src_hit, tgt_hit, total = 0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for img, lbl in pbar:
        img, lbl = img.to(device), lbl.to(device)

        # 1. --- 核心修改：初筛（过滤掉原始模型就认错的图） ---
        with torch.no_grad():
            clean_pred = src_model(img).max(1)[1]
            # 找到预测正确的索引掩码 (mask)
            mask = (clean_pred == lbl)

            # 如果这一批次里所有图都被认错了，直接跳过
            if mask.sum() == 0:
                continue

            # 只保留认对的样本
            img = img[mask]
            lbl = lbl[mask]

        # 2. --- 只针对认对的样本生成对抗样本 ---
        adv = attacker(img, lbl)

        # 3. --- 统计攻击效果 ---
        with torch.no_grad():
            # 此时在 Eps=0 时，src_model(adv) 必然等于 lbl
            src_hit += (src_model(adv).max(1)[1] != lbl).sum().item()
            tgt_hit += (tgt_model(adv).max(1)[1] != lbl).sum().item()

        # 这里的 total 只累加被保留的“高质量样本”数量
        total += lbl.size(0)

        pbar.set_postfix({
            "Src": f"{(src_hit / total) * 100:.1f}%",
            "Tgt": f"{(tgt_hit / total) * 100:.1f}%"
        })

    return (src_hit / total) * 100, (tgt_hit / total) * 100