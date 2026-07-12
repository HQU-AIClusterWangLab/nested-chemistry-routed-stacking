"""
Phase 0.1：读入全部 .pt 文件，打印 dataset summary
"""
import os
import torch
from collections import defaultdict

DATASET_DIR = r'D:\lunwen\2.1sci\phase 0\dataset\processed'

def main():
    if not os.path.exists(DATASET_DIR):
        print(f'目录不存在: {DATASET_DIR}')
        return

    all_samples = []
    system_stats = defaultdict(lambda: {'count': 0, 'groups': set(), 'has_forces': True, 'sizes': set()})
    
    # 遍历所有 system 子目录
    for system_id in sorted(os.listdir(DATASET_DIR)):
        system_dir = os.path.join(DATASET_DIR, system_id)
        if not os.path.isdir(system_dir):
            continue
        
        for fname in os.listdir(system_dir):
            if not fname.endswith('.pt'):
                continue
            fpath = os.path.join(system_dir, fname)
            try:
                data = torch.load(fpath, weights_only=False)
            except Exception as e:
                print(f'  警告: 读取 {fpath} 失败: {e}')
                continue
            
            # 基本信息
            sid = getattr(data, 'system_id', system_id) if hasattr(data, 'system_id') else system_id
            gid = getattr(data, 'group_id', -1) if hasattr(data, 'group_id') else -1
            tid = getattr(data, 'trajectory_id', -1) if hasattr(data, 'trajectory_id') else -1
            iid = getattr(data, 'init_structure_id', -1) if hasattr(data, 'init_structure_id') else -1
            
            has_force = hasattr(data, 'forces') and data.forces is not None and data.forces.numel() > 0
            
            if hasattr(data, 'x'):
                n_atoms = data.x.shape[0]
            else:
                n_atoms = -1
            
            system_stats[sid]['count'] += 1
            system_stats[sid]['groups'].add(gid)
            system_stats[sid]['sizes'].add(n_atoms)
            if not has_force:
                system_stats[sid]['has_forces'] = False
            
            all_samples.append({
                'system_id': sid,
                'group_id': gid,
                'trajectory_id': tid,
                'init_structure_id': iid,
                'n_atoms': n_atoms,
                'has_forces': has_force,
            })

    # ========== 打印 Summary ==========
    print('=' * 70)
    print('PHASE 0.1 - DATASET SUMMARY')
    print('=' * 70)

    total = len(all_samples)
    print(f'\n总样本数: {total}')
    print(f'系统数量: {len(system_stats)}')
    print(f'系统列表: {sorted(system_stats.keys())}')

    all_have_forces = all(s['has_forces'] for s in system_stats.values())
    print(f'所有体系都有 forces: {all_have_forces}')

    print(f'\n--- 每个体系的统计 ---')
    print(f'{"System":<15} {"Samples":>8} {"Groups":>8} {"Sizes":>10} {"Forces":>8}')
    print('-' * 55)
    for sid in sorted(system_stats.keys()):
        st = system_stats[sid]
        sizes_str = str(sorted(st['sizes']))
        if len(sizes_str) > 25:
            sizes_str = sizes_str[:22] + '...'
        print(f'{sid:<15} {st["count"]:>8} {len(st["groups"]):>8} {sizes_str:>10} {str(st["has_forces"]):>8}')

    # 额外检查
    print(f'\n--- 全局检查 ---')
    
    # 检查 group_id 分布
    group_dist = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        group_dist[s['system_id']][s['group_id']] += 1
    
    for sid in sorted(group_dist.keys()):
        groups = group_dist[sid]
        min_g, max_g = min(groups.values()), max(groups.values())
        print(f'  {sid}: group 内样本数范围 {min_g} ~ {max_g}')

    # 检查是否有 group_id = -1（未解析）
    missing_groups = sum(1 for s in all_samples if s['group_id'] == -1)
    if missing_groups > 0:
        print(f'\n⚠ 警告: {missing_groups} 个样本的 group_id 未成功解析 (-1)')

    print(f'\n{"=" * 70}')
    print('Phase 0.1 完成。')
    print(f'{"=" * 70}')
    
    return all_samples, system_stats


if __name__ == '__main__':
    main()
