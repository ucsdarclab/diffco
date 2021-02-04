from abc import abstractclassmethod
import json
from os.path import join, basename
import numpy as np


# !!Use after BACKUP the files being written!!
# loadexps = ['results/2d_3dof_exp1']
saveexps = ['results/2d_7dof_exp2']
methods = ['fclgradfree']

## ========== Recover missing exp result from previous result files. =============
# for saveexp, loadexp in zip(saveexps, loadexps):
#     from glob import glob
#     envs = sorted(glob(join(saveexp, '2d*.json'),))
#     for env in envs:
#         with open(join(loadexp, basename(env)), 'r') as loadfp:
#             loadrec = json.load(loadfp)
#         with open(join(saveexp, basename(env)), 'r') as savefp:
#             saverec = json.load(savefp)
#         for m in methods:
#             assert m in loadrec
#             saverec[m] = loadrec[m]
#         with open(join(saveexp, basename(env)), 'w') as savefp:
#             json.dump(saverec, savefp, indent=4)
#         print('Written: {}'.format(env))

        # break # DEBUG
# =================================================================================

## =========== Remove wrong number of repair results of FCLGRADFREE =================
# 1.verify it is all zero
for saveexp in saveexps:
    from glob import glob
    envs = sorted(glob(join(saveexp, '2d*.json'),))
    for env in envs:
        with open(join(saveexp, basename(env)), 'r') as savefp:
            saverec = json.load(savefp)
        for m in methods:
            assert m in saverec
            l = len(saverec[m]['repair_cnt_check'])
            print(env, l)
            if l < 10: continue
            if l == 12:
                indices = [-12, -10] + list(range(-8, 0))
            elif l == 20:
                indices = list(range(0, 20, 2))
            else:
                indices = list(range(10))
            for i, idx in enumerate(indices):
                assert saverec[m]['repair_cnt_check'][idx] == 0
                saverec[m]['repair_cnt_check'][i] = 0
                assert saverec[m]['repair_success'][idx] == saverec[m]['success'][i]
                saverec[m]['repair_success'][i] = saverec[m]['success'][i]
                assert saverec[m]['repair_time'][idx] == 0
                saverec[m]['repair_time'][i] = 0
                assert saverec[m]['repair_cost'][idx] == saverec[m]['cost'][i]
                saverec[m]['repair_cost'][i] = saverec[m]['cost'][i]
                assert np.all(np.array(saverec[m]['repair_solution'][idx]) == np.array(saverec[m]['solution'][i]))
                saverec[m]['repair_solution'][i] = saverec[m]['solution'][i]
            saverec[m]['repair_cnt_check'] = saverec[m]['repair_cnt_check'][:10]
            assert len(saverec[m]['repair_cnt_check']) == 10
            saverec[m]['repair_success'] = saverec[m]['repair_success'][:10]
            assert len(saverec[m]['repair_success']) == 10
            saverec[m]['repair_time'] = saverec[m]['repair_time'][:10]
            assert len(saverec[m]['repair_time']) == 10
            saverec[m]['repair_cost'] = saverec[m]['repair_cost'][:10]
            assert len(saverec[m]['repair_cost']) == 10
            saverec[m]['repair_solution'] = saverec[m]['repair_solution'][:10]
            assert len(saverec[m]['repair_solution']) == 10
        with open(join(saveexp, basename(env)), 'w') as savefp:
            json.dump(saverec, savefp, indent=4)
            print('Written: {}'.format(env))

        # break # DEBUG
