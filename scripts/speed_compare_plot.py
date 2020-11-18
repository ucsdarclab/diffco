import numpy as np
from glob import glob
import json
from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns
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

methods = ['diffco', 'givengrad', 'bidiffco', 'fclgradfree']
keys = ['success', 'time', 'cnt_check', 'cost']
obsnums = [1, 2, 5, 10, 20]
dofs = ['{}dof'.format(d) for d in [2, 3, 7]]
stats_by_obsnum = {k: {d: {n: {m: [] for m in methods}
                           for n in obsnums} for d in dofs} for k in keys}

exps = ['2d_2dof_exp1', '2d_3dof_exp1', '2d_7dof_exp1']
for exp_name, dof in zip(exps, dofs):
    for n in obsnums:
        for m in methods:
            for fn in sorted(glob('results/'+exp_name+'/*_{}obs_*.json'.format(n))):
                if not '{}obs'.format(n) in fn:
                    continue
                with open(fn, 'r') as f:
                    r = json.load(f)
                # if m not in r:
                #     continue
                for k in keys:
                    stats_by_obsnum[k][dof][n][m].append(r[m][k])
            for k in keys:
                stats_by_obsnum[k][dof][n][m] = np.concatenate(
                    stats_by_obsnum[k][dof][n][m])

# for m in methods:
#     for n in obsnums:
#         for k in keys:
#             if stats_by_obsnum[k][n][m] == []:
#                 print(n, m, k)
#                 continue
#             stats_by_obsnum[k][n][m] = np.concatenate(stats_by_obsnum[k][n][m])

plot_dof_groups = [dofs] #[[dof] for dof in dofs]  # could be [dofs] to merge dofs

w = 0.3
x = np.arange(len(obsnums))*1.5
fig = plt.figure(figsize=(3*len(keys), 3*len(plot_dof_groups)))
methods_labeltext = {
    'diffco': 'DiffCo+Adam',
    'givengrad': 'DiffCo+SLSQP',
    'bidiffco': 'Binary DiffCo+SLSQP',
    'fclgradfree': 'FCL+SLSQP',
}
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
            tmp_datas = {n: np.concatenate([stats_by_obsnum[k][dof][n][m] if k != 'cost'
                                            else stats_by_obsnum[k][dof][n][m][stats_by_obsnum['success'][dof][n][m]] for dof in dof_group])
                         for n in obsnums}
            mean_val = np.array([np.mean(tmp_datas[n]) for n in obsnums])
            if all([len(stats_by_obsnum['success'][dof][n][m]) > 0 for dof in dof_group]):
                min_val = np.array([np.min(tmp_datas[n]) for n in obsnums])
                max_val = np.array([np.max(tmp_datas[n]) for n in obsnums])
                # if k == 'cost':
                #     print(mean_val)
                #     print(min_val)
                #     print(max_val)
                #     print('')
            else:
                print('empty list: ', dof_group, m)
            ax.bar(x+itm*w, mean_val, width=w, yerr=None if k == 'success'
                   else (mean_val-min_val, max_val-mean_val), 
                   error_kw=dict(lw=1.2, capsize=3, capthick=0.8, ecolor='darkgray'), label=methods_labeltext[m])

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
                    ncol=len(keys), bbox_to_anchor=(0.5, -0.06/len(plot_dof_groups)), borderaxespad=0)

plt.tight_layout()
# plt.show()
plt.savefig('figs/speed_compare/stats_by_obsnum.pdf',
            dpi=500, bbox_inches='tight')
