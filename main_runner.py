import argparse
import yaml
import torch
import os
from torchvision import datasets, transforms
from core.utils import get_models
import task_01_transferability.run_grid_search as exp_grid
import task_01_transferability.run_saturation as exp_sat
from task_01_transferability.visualizer import plot_results


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="AI Security Experiment Runner")
    parser.add_argument('--config', type=str, default='all', help="Path to config file or 'all'")
    args = parser.parse_args()

    # 1. 初始化设备与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model, tgt_model = get_models(device)

    # 2. 标准 ImageNet 预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # 直接使用标准 ImageFolder，因为 utils.py 会负责筛选认对的样本
    dataset = datasets.ImageFolder(root="data/val", transform=transform)

    # 3. 任务列表定义
    if args.config == 'all':
        tasks = ["configs/grid_search.yaml", "configs/saturation.yaml", "configs/saturation_refine.yaml"]
    else:
        tasks = [args.config]

    for config_path in tasks:
        if not os.path.exists(config_path):
            continue

        cfg = load_yaml(config_path)

        # 4. 配置参数预处理（归一化到 0-1 空间）
        if 'alpha' not in cfg:
            cfg['alpha'] = 2.0 / 255.0
        else:
            cfg['alpha'] = cfg['alpha'] / 255.0 if cfg['alpha'] >= 1 else cfg['alpha']

        cfg['eps_list'] = [e / 255.0 if e >= 1 else e for e in cfg['eps_list']]

        # 5. 结果目录准备
        result_csv = cfg['save_path']
        result_dir = os.path.dirname(result_csv)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print(f"\n" + "=" * 50)
        print(f"[*] 正在运行任务: {cfg['name']} | 设备: {device}")

        # 6. 执行实验逻辑
        if os.path.exists(result_csv) and os.path.getsize(result_csv) > 100:
            print(f"    [!] 检测到结果已存在，正在执行可视化更新...")
        else:
            print(f"    [>] 启动攻击核心逻辑 (初筛模式)...")
            if cfg['name'] == "grid_search":
                exp_grid.run(cfg, dataset, src_model, tgt_model, device)
            elif cfg['name'] in ["saturation", "saturation_refine"]:
                exp_sat.run(cfg, dataset, src_model, tgt_model, device)

        # 7. 可视化渲染
        is_grid = (cfg['name'] == "grid_search")
        plot_results(result_csv, result_dir, is_grid=is_grid, filename_suffix=cfg['name'])

    print(f"\n✅ 所有实验任务已按计划完成！")


if __name__ == "__main__":
    main()