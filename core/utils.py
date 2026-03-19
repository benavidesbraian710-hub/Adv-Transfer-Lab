import torch
from torchvision import models

def get_models(device):
    """加载并冻结 ResNet18 (Source) 和 VGG16 (Target)"""
    src = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    tgt = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    for p in src.parameters(): p.requires_grad = False
    for p in tgt.parameters(): p.requires_grad = False
    return src, tgt


from tqdm import tqdm  # 确保顶部导入了 tqdm


def run_attack_batch(loader, attacker, src_model, tgt_model, device, desc="Attacking"):
    src_hit, tgt_hit, total = 0, 0, 0
    # 将 tqdm 绑定到 loader 上，实现 5000 样本的实时进度
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for img, lbl in pbar:
        img, lbl = img.to(device), lbl.to(device)
        adv = attacker(img, lbl)
        with torch.no_grad():
            src_hit += (src_model(adv).max(1)[1] != lbl).sum().item()
            tgt_hit += (tgt_model(adv).max(1)[1] != lbl).sum().item()
        total += lbl.size(0)

        # 实时在进度条右侧刷新当前的 ASR
        pbar.set_postfix({"Src": f"{(src_hit / total) * 100:.1f}%", "Tgt": f"{(tgt_hit / total) * 100:.1f}%"})

    return (src_hit / total) * 100, (tgt_hit / total) * 100