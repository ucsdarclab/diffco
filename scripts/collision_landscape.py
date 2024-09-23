
import os
import json
from time import time
from tqdm import tqdm

import numpy as np
import torch

from diffco import DiffCo, MultiDiffCo
from diffco import kernel
from diffco import utils
from diffco import FCLChecker

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
# import seaborn as sns
# sns.set()
import matplotlib.patheffects as path_effects

from escape import *

def create_plots(dist_list, cfgs=None, labels=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    if labels is None:
        labels = [None] * len(dist_list)
    if cfgs is None:
        cfgs = torch.linspace(-np.pi, np.pi, len(dist_list[0]))
    fig, ax = plt.subplots(figsize=(8, 3))

    color = 'tab:red'
    ax.plot(cfgs, dist_list[0], color=color)
    ax.set_xlabel('Joint 1')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=10)
    ax.axhline(0, linestyle='--', color='gray', alpha=0.5)

    ax.set_ylabel(labels[0], color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(-dist_list[0].abs().max()*1.1, dist_list[0].abs().max()*1.1)
    # ax.legend()
    if len(dist_list) > 1:
        ax = ax.twinx()
        color = 'tab:blue'
        ax.plot(cfgs, dist_list[1], label=labels[1], color=color)
        ax.set_ylabel(labels[1], color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(-dist_list[1].abs().max()*1.1, dist_list[1].abs().max()*1.1)
        # ax.legend()
        
    # plt.legend()

# Commented out lines include convenient code for debugging purposes
def main(datapath, test_num=360, key=None):
    env_name = os.path.splitext(os.path.basename(datapath))[0]
    if key is not None:
        env_name = f'{env_name}_{key}'
    res_dir = 'results/collision_landscape'
    os.makedirs(res_dir, exist_ok=True)
    resfilename = os.path.join(res_dir, ''f'landscape_{env_name}.json')

    # =========================================================
    dataset = torch.load(datapath)
    cfgs = dataset['data']
    labels = dataset['label']
    dists = dataset['dist']
    obstacles = dataset['obs']
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine

    checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check DiffCo test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.9)

    fitting_target = 'dist' # {label, dist, hypo}
    Epsilon = 1 #0.01
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target)
    dist_est = checker.rbf_score
    min_score = dist_est(cfgs[train_num:]).min().item()
    print('MIN_SCORE = {:.6f}'.format(min_score))

    fcl_checker = FCLChecker(obstacles, robot, label_type='binary')

    
    j1 = torch.linspace(-np.pi, np.pi, test_num)
    test_cfgs = torch.zeros(test_num, robot.dof)
    test_cfgs[:, 0] = j1
    diffco_dist = dist_est(test_cfgs)
    fcl_dist = fcl_checker(test_cfgs, distance=True)[1]

    with open(resfilename, 'w') as f:
        json.dump({
            'dof': robot.dof,
            'cfgs': test_cfgs.tolist(),
            'diffco_dist': diffco_dist.tolist(),
            'fcl_dist': fcl_dist.tolist()
        }, f, indent=1)
        print('Distance estimations recorded in {}'.format(f.name))
    # ====================================================================

    ## This is for loading previously computed trajectory
    with open(resfilename, 'r') as f:
        rec = json.load(f)
        diffco_dist = torch.FloatTensor(rec['diffco_dist'])
        fcl_dist = torch.FloatTensor(rec['fcl_dist'])
        cfgs = torch.FloatTensor(rec['cfgs'])
        dof = rec['dof']
    
    create_plots([fcl_dist, diffco_dist], cfgs=cfgs[:, 0], labels=['FCL Distance', 'DiffCo Collision Score'])
    # create_plots([fcl_dist], labels=['FCL'])


    # (Recommended) This produces a single shot of the valid cfgs
    # single_plot(robot, torch.FloatTensor(valid_cfg_rec['cfgs']), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
    plt.tight_layout()
    fig_dir = 'figs/collision_landscape'
    os.makedirs(fig_dir, exist_ok=True)
    # plt.show()
    plt.savefig(os.path.join(fig_dir, '2d_{}dof_{}.pdf'.format(dof, env_name)))
    
    




if __name__ == "__main__":
    main('data/landscape/2d_3dof_150obs_binary_3d_halfnarrow.pt') #'equalmargin'
    # main('data/compare_escape/2d_7dof_300obs_binary_7d_narrow.pt', 1000, key='resampling') #'equalmargin'
    # main('2class_1', 3, key='equalmargin')