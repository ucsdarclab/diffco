import json
import torch
import os
from os.path import basename, splitext, join, isdir
import sys
sys.path.append('/home/yuheng/DiffCo/')
from diffco import DiffCo, MultiDiffCo

if __name__ == "__main__":
    exp_name = '2d_7dof_exp1'
    folder = join('data', exp_name)
    from glob import glob
    envs = sorted(glob(join(folder, '*.pt'),))

    for env_name in envs:
        dataset = torch.load(env_name)
        cfgs = dataset['data']
        labels = dataset['label']
        free_cfgs = cfgs[labels == -1]
        s_cfgs = []
        t_cfgs = []
        for test_it in range(100):
            indices = torch.randint(0, len(free_cfgs), (2, ))
            while indices[0] == indices[1]:
                indices = torch.randint(0, len(free_cfgs), (2, ))
            print('Config indices: ', indices)
            start_cfg = free_cfgs[indices[0]] 
            target_cfg = free_cfgs[indices[1]]
            s_cfgs.append(start_cfg)
            t_cfgs.append(target_cfg)
        with open(splitext(env_name)[0]+'_testcfgs.json', 'w') as f:
            json.dump({
                'env_name': splitext(basename(env_name))[0],
                'start_cfgs': torch.stack(s_cfgs).numpy().tolist(),
                'target_cfgs': torch.stack(t_cfgs).numpy().tolist(),
            }, f, indent=1)

    
