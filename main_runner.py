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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model, tgt_model = get_models(device)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root="data/val", transform=transform)

    if args.config == 'all':
        tasks = [
            "configs/grid_search.yaml",
            "configs/saturation.yaml",
            "configs/saturation_refine.yaml"
        ]
    else:
        tasks = [args.config]

    for config_path in tasks:
        if not os.path.exists(config_path):
            if args.config != 'all':
                print(f"[!] 错误：配置文件 {config_path} 不存在。")
            continue

        cfg = load_yaml(config_path)

        # 修改点：Alpha 处理逻辑
        # 优先读 YAML，若无则使用之前的基准 2/255
        if 'alpha' not in cfg:
            cfg['alpha'] = 2.0 / 255.0
        else:
            cfg['alpha'] = cfg['alpha'] / 255.0 if cfg['alpha'] >= 1 else cfg['alpha']

        cfg['eps_list'] = [e / 255.0 if e >= 1 else e for e in cfg['eps_list']]

        result_csv = cfg['save_path']
        result_dir = os.path.dirname(result_csv)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print(f"\n" + "=" * 50)
        print(f"[*] 处理任务: {cfg['name']} | Alpha: {cfg['alpha']:.5f}")

        if os.path.exists(result_csv):
            print(f"    [!] 检测到 CSV 数据已存在。跳过攻击，更新图表...")
        else:
            print(f"    [>] 启动攻击实验 (Alpha={cfg['alpha']:.5f})...")
            if cfg['name'] == "grid_search":
                exp_grid.run(cfg, dataset, src_model, tgt_model, device)
            elif cfg['name'] in ["saturation", "saturation_refine"]:
                exp_sat.run(cfg, dataset, src_model, tgt_model, device)
            else:
                print(f"    [!] 未知任务类型 '{cfg['name']}'。")
                continue

        # 修改点：绘图时传入 filename_suffix 参数
        print(f"    [>] 正在导出可视化结果至: {result_dir}")
        is_grid = (cfg['name'] == "grid_search")
        plot_results(result_csv, result_dir, is_grid=is_grid, filename_suffix=cfg['name'])

        print(f"✅ 任务 {cfg['name']} 处理完成！")


if __name__ == "__main__":
    main()