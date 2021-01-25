import numpy as np
from glob import glob
import json
from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from torch._C import dtype
sns.set()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'sans-serif',  # 'sans-serif', #  'Microsoft Sans Serif' "Times New Roman"
})  # "font.sans-serif": ["Helvetica"]})

# print(font_manager.get_cachedir(), font_manager._fmcache)

# methods = ['diffco', 'givengrad', 'bidiffco', 'fclgradfree']
# keys = ['success', 'time', 'cnt_check', 'cost']
# stats_by_method = {k: {m: [] for m in methods} for k in keys}

# exps = ['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1']
# for exp_name in exps:
#     for fn in sorted(glob('results/'+exp_name+'/*.json')):
#         # if not '10obs' in fn:
#         #     continue
#         with open(fn, 'r') as f:
#             r = json.load(f)
#         for m in methods:
#             if m not in r:
#                 continue
#             for k in keys:
#                 stats_by_method[k][m].append(r[m][k])
# for m in methods:
#     for k in keys:
#         if stats_by_method[k][m] == []:
#             continue
#         stats_by_method[k][m] = np.concatenate(stats_by_method[k][m])

# fig = plt.figure(figsize=(8, 10))
# for i, k in enumerate(keys, 1):
#     ax  = fig.add_subplot(2, len(keys)//2, i)
#     ax.bar(methods, [np.mean(stats_by_method[k][m]) for m in methods])#, yerr=[np.std(stats_by_method[k][m]) for m in methods], capsize=10)
#     ax.set_title(k.replace('_', '\_'))

# plt.show()

methods = ['givengrad', 'margindiffcogradfree', 'fcldist'] #'fclgradfree', 'bidiffco', 'diffco',  'diffcogradfree',  ['diffco', 'givengrad', 'bidiffco', 'fclgradfree']
keys = ['success', 'time', 'cnt_check', 'cost']
repair_keys = ['repair_'+k for k in keys]
val_keys = ['val_time']
sanity_check_keys = ['solution', 'repair_solution']
all_keys = keys+repair_keys+val_keys+sanity_check_keys
obsnums = [1, 2, 5, 10, 20]
dofs = ['{}dof'.format(d) for d in [2, 3, 7]] * 2
stats_by_obsnum = {k: {d: {n: {m: [] for m in methods}
                           for n in obsnums} for d in dofs} for k in all_keys}

# exps = ['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1']
exps = ['2d_2dof_exp2', '2d_3dof_exp2', '2d_7dof_exp2', '2d_2dof_exp3', '2d_3dof_exp3', '2d_7dof_exp3'] #'2d_2dof_exp3', '2d_3dof_exp3', '2d_7dof_exp3'] #
for exp_name, dof in zip(exps, dofs):
    for n in obsnums:
        for fn in sorted(glob('results/'+exp_name+'/*_{}obs_*.json'.format(n))):
            if not '{}obs'.format(n) in fn:
                continue
            with open(fn, 'r') as f:
                r = json.load(f)
            for m in r: 
                if m not in methods:
                    continue
                for k in all_keys:
                    stats_by_obsnum[k][dof][n][m].append(r[m][k])

for k in all_keys:
    for dof in stats_by_obsnum[k]:
        for n in stats_by_obsnum[k][dof]:
            for m in stats_by_obsnum[k][dof][n]:
                if len(stats_by_obsnum[k][dof][n][m]) == 0:
                    print('Missing: ', k, dof, n, m)
                    continue
                stats_by_obsnum[k][dof][n][m] = np.concatenate(stats_by_obsnum[k][dof][n][m])

# Sanity Check
for dof in stats_by_obsnum['solution']:
    for n in stats_by_obsnum['solution'][dof]:
        if len(stats_by_obsnum['solution'][dof][n]['margindiffcogradfree']) == 0: # fcldist
            continue
        s = stats_by_obsnum['solution'][dof][n]['margindiffcogradfree'][:, [0, -1]]
        # print('margindiffcogradfree', s.shape)
        # print('fcldist', s.shape)
        for m in stats_by_obsnum['solution'][dof][n]:
            # if m == 'fclgradfree' or m == 'margindiffcogradfree' or m == 'fcldist':
            #     continue
            # if m == 'fclgradfree' and '7' in dof and n in [2, 5]:
            #     continue
            if m == 'fcldist' and len(stats_by_obsnum['solution'][dof][n][m]) == 0:
                continue
            assert type(stats_by_obsnum['solution'][dof][n][m]) == np.ndarray, \
                (m, type(stats_by_obsnum['solution'][dof][n][m]), stats_by_obsnum['solution'][dof][n][m])
            sprime = stats_by_obsnum['repair_solution'][dof][n][m][:, [0, -1]]
            # if m == 'fclgradfree' and '7' in dof and n in [1, 2, 10, 20]:
            #     sprime = sprime[-120:]
            #     sprime = sprime.reshape((10, 12, 2, 7))
            #     sprime = sprime[:, [0, 2]+list(range(4, 12))].reshape((-1, 2, 7))
            if not (s.shape == sprime.shape and np.all(s == sprime)): #, 
                print('Error::::', dof, n, m, s.shape, sprime.shape) #, '\n', s[:, :, 0]-sprime[:, :, 0], '\n') #, s if s.shape == sprime.shape else [s.shape, sprime.shape])

print('Sanity check finished.')
# exit(1)

plot_dof_groups = [[dof] for dof in list(stats_by_obsnum[keys[0]].keys())] # [dofs] #[[dof] for dof in dofs]  # could be [dofs] to merge dofs

w = 0.5
x = np.arange(len(obsnums))*w*(len(methods) + 2)
fig = plt.figure(figsize=(3*len(keys), 3*len(plot_dof_groups)))
methods_labeltext = {
    'diffco': 'DiffCo+Adam',
    'givengrad': 'DiffCo+SLSQP',
    'bidiffco': 'BinDiffCo+SLSQP',
    'diffcogradfree': 'NondiffCo+SLSQP',
    'margindiffcogradfree': 'NondiffCo+SLSQP',  #margin 
    'fclgradfree': 'FCLBin+SLSQP',
    'fcldist': 'FCL+SLSQP',
}
sns.set_palette(sns.color_palette("Paired"))
# keylabeltext = {
#     'success': '(a) Success Rate',
#     'time': '(b) Time (s)',
#     'cnt_check': '(c) No. of Collision Checks',
#     'cost': '(d) Cost'
# }
keylabeltext = {
    'success': 'Success Rate',
    'time': 'Time (s)',
    'cnt_check': 'No. of Collision Checks',
    'cost': 'Cost'
}
from string import ascii_lowercase

for j, dof_group in enumerate(plot_dof_groups):
    for i, k in enumerate(keys, 1):
        ax = fig.add_subplot(len(plot_dof_groups), len(keys), j*len(keys)+i)
        ax.grid(True, 'both')
        for itm, m in enumerate(methods):
            # Only count costs of successful plans. Unsuccessful ones do not make sense.
            tmp_datas = {}
            tmp_repair_datas = {}
            for n in obsnums:
                dlist = []
                rlist = []
                for dof in dof_group:
                    if len(stats_by_obsnum[k][dof][n][m]) == 0:
                        continue
                    if k == 'cost':
                        dlist.append(stats_by_obsnum[k][dof][n][m][stats_by_obsnum['success'][dof][n][m]])
                        rlist.append(stats_by_obsnum['repair_'+k][dof][n][m][stats_by_obsnum['repair_'+'success'][dof][n][m]])
                    elif k == 'time':
                        dlist.append(stats_by_obsnum[k][dof][n][m]+stats_by_obsnum['val_'+k][dof][n][m])
                        rlist.append(stats_by_obsnum['repair_'+k][dof][n][m])
                    else:
                        dlist.append(stats_by_obsnum[k][dof][n][m])
                        rlist.append(stats_by_obsnum['repair_'+k][dof][n][m])
                if dlist == []:
                    tmp_datas[n] = dlist
                    tmp_repair_datas[n] = rlist
                    continue
                tmp_datas[n] = np.concatenate(dlist)
                tmp_repair_datas[n] = np.concatenate(rlist)

            # Original confusing way of coding
            # tmp_datas = {n: np.concatenate([stats_by_obsnum[k][dof][n][m] if k != 'cost'
            #                                 else stats_by_obsnum[k][dof][n][m][stats_by_obsnum['success'][dof][n][m]] for dof in dof_group])
            #              for n in obsnums}
            # tmp_repair_datas = {n: np.concatenate([stats_by_obsnum[k][dof][n]['repair_'+m] if k != 'cost'
            #                                         else stats_by_obsnum[k][dof][n]['repair_'+m][stats_by_obsnum['success'][dof][n]['repair_'+m]] for dof in dof_group])
            #                     for n in obsnums}

            mean_val = np.array([np.mean(tmp_datas[n]) for n in obsnums])
            mean_repair_val = np.array([np.mean(tmp_repair_datas[n]) for n in obsnums])
            if k in ['time', 'cnt_check']:
                mean_repair_val += mean_val
            if all([len(stats_by_obsnum['success'][dof][n][m]) > 0 for dof in dof_group for n in stats_by_obsnum['success'][dof]]):
                min_val = np.array([np.min(tmp_datas[n]) for n in obsnums], dtype=np.float64)
                max_val = np.array([np.max(tmp_datas[n]) for n in obsnums], dtype=np.float64)
                min_repair = np.array([np.min(tmp_repair_datas[n]) for n in obsnums], dtype=np.float64)
                max_repair = np.array([np.max(tmp_repair_datas[n]) for n in obsnums], dtype=np.float64)
                if k in ['time', 'cnt_check']:
                    min_repair += mean_val
                    max_repair += mean_val
                # if k == 'cost':
                #     print(mean_val)
                #     print(min_val)
                #     print(max_val)
                #     print('')
            else:
                print('empty list: ', dof_group, m)
                min_val = mean_val
                max_val = mean_val
                min_repair = mean_repair_val
                max_repair = mean_repair_val
            # ax.bar(x+itm*w, mean_val if k != 'cost' else 0, bottom=mean_val, width=w, yerr=None if True #k == 'success' #mean_repair_val if k == 'cost' else mean_val
            #        else (mean_repair_val-min_repair, max_repair-mean_repair_val), 
            #        error_kw=dict(lw=0.5, capsize=2, capthick=0.8, ecolor='darkgray'), label=methods_labeltext[m]+' (repair stage)',
            #        #alpha=0.7
            #        ) # Just checking the validity of the log-scale graphs: total value should be twice the value of non-repair part
            ax.bar(x+itm*w, mean_repair_val-mean_val if k != 'cost' else 0, bottom=mean_val, width=w, yerr=None if True #k == 'success' #mean_repair_val if k == 'cost' else mean_val
                   else (mean_repair_val-min_repair, max_repair-mean_repair_val), 
                   error_kw=dict(lw=0.5, capsize=2, capthick=0.8, ecolor='darkgray'), label=methods_labeltext[m]+' (repair stage)',
                   #alpha=0.7
                   )
            ax.bar(x+itm*w, mean_val, width=w, yerr=None if k == 'success'
                   else (mean_val-min_val, max_val-mean_val), 
                   error_kw=dict(lw=0.5, capsize=3, capthick=1.5, ecolor='darkgray'), label=methods_labeltext[m])
            

        if len(plot_dof_groups) > 1:
            ax.set_title("({}) {} - {}".format(
                ascii_lowercase[j*len(keys)+i-1],
                keylabeltext[k],
                ','.join(dof_group).replace('dof','')+'DOF', ))  # labelpad=-5, loc='top')
        else:
            ax.set_title("({}) {}".format(
                ascii_lowercase[j*len(keys)+i-1],
                keylabeltext[k],))  # labelpad=-5, loc='top')
        ax.set_xticks(x + len(methods)/2*w)
        ax.set_xticklabels(['{}obs'.format(n) for n in obsnums])
        ax.tick_params(axis='y', which='major', pad=-5)
        ax.tick_params(axis='x', which='major', pad=-4)
        if k not in ['success']:
            ax.set_yscale('log')
    if j == len(plot_dof_groups)-1:
        # ax.legend(bbox_to_anchor=(1.300, 1), loc="upper right", ncol=1, fontsize=5) #
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                    ncol=len(methods), bbox_to_anchor=(0.5, -0.2/len(plot_dof_groups)), borderaxespad=1)
plt.tight_layout()
# plt.show()
plt.savefig('figs/speed_compare/stats_by_obsnum_repaired_{}.pdf'.format("_".join(methods)),
            dpi=500, bbox_inches='tight')
