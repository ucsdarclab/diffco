import torch
from glob import glob
import sys
# sys.path.append('/home/yuheng/DiffCo/')
from diffco import DiffCo


folders = ['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1']

for folder in folders:
    for fn in glob('data/'+folder+'/*.pt'):
        dataset = torch.load(fn)
        if torch.sum(dataset['label'] == 1) < 20:
            print(fn, torch.sum(dataset['label'] == 1))