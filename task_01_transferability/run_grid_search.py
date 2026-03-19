import os, csv, torch, itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from core.utils import run_attack_batch
from torchattacks import PGD

def run(config, dataset, src_model, tgt_model, device):
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    tasks = list(itertools.product(config['eps_list'], config['step_list']))

    with open(config['save_path'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["eps", "step", "src_mean", "src_std", "tgt_mean", "tgt_std"])

    pbar = tqdm(tasks, desc=f"🔍 {config['name']}", dynamic_ncols=True)
    for eps, step in pbar:
        s_list, t_list = [], []
        for t in range(config['trials']):
            pbar.set_description(f"Grid: Eps {eps * 255:.1f} | Step {step} | T {t + 1}")
            idx = np.random.choice(len(dataset), config['num_samples'], replace=False)
            loader = DataLoader(Subset(dataset, idx), batch_size=config['batch_size'])

            # 修改点：从 config 中获取 alpha
            attacker = PGD(src_model, eps=eps, alpha=config['alpha'], steps=step)
            current_desc = f"Eps {round(eps * 255)}/255 | Step {step}"
            s, t_res = run_attack_batch(loader, attacker, src_model, tgt_model, device, desc=current_desc)
            s_list.append(s)
            t_list.append(t_res)

        with open(config['save_path'], 'a', newline='') as f:
            csv.writer(f).writerow([eps, step, np.mean(s_list), np.std(s_list), np.mean(t_list), np.std(t_list)])
        pbar.set_postfix({"Src_ASR": f"{np.mean(s_list):.1f}%", "Tgt_ASR": f"{np.mean(t_list):.1f}%"})