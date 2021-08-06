from os import makedirs, pipe
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
    "font.family": 'sans-serif',  
})

# methods = ['givengrad', 'fcldist'] #'fclgradfree', 'bidiffco', 'diffco',  'diffcogradfree',  ['diffco', 'givengrad', 'bidiffco', 'fclgradfree']
class Pipeline:
    def __init__(self, use_plan, checker, optim, labeltxt):
        self.use_plan = use_plan
        self.checker = checker
        self.optim = optim
        self.labeltxt = labeltxt
    
    def get_value(self, value_name, data_record) -> np.ndarray:
        # This deals with any logic when computing values
        if value_name == 'success':
            return np.array(data_record[value_name])
        elif value_name == 'time':
            if self.use_plan and self.optim is None:
                return np.array(data_record['plan_time']) + np.array(data_record['val_time'])
            elif self.use_plan and self.optim is not None:
                return np.array(data_record['plan_time'])+np.array([data_record['optim_time']])+np.array([data_record['val_time']])
            elif not self.use_plan and self.optim is not None:
                return np.array([data_record['optim_time']])+np.array([data_record['val_time']])
            else:
                raise NotImplementedError('This pipeline case has not been covered.')
        elif value_name == 'cnt_check': 
            if self.use_plan and self.optim is None:
                return np.array(data_record['plan_cnt_check'])
            elif self.use_plan and self.optim is not None:
                return np.array(data_record['plan_cnt_check'])+np.array([data_record['optim_cnt_check']])
            elif not self.use_plan and self.optim is not None:
                return np.array([data_record['optim_cnt_check']])
            else:
                raise NotImplementedError('This pipeline case has not been covered.')
        elif value_name == 'cost':
            success = np.array(data_record['success'])
            if self.use_plan and self.optim is None:
                ret = np.array(data_record['plan_cost'])[success].astype(float)
            elif self.use_plan and self.optim is not None:
                ret = np.array(data_record['optim_cost'])[success].astype(float)
            elif not self.use_plan and self.optim is not None:
                ret = np.array(data_record['optim_cost'])[success].astype(float)
            else:
                raise NotImplementedError('This pipeline case has not been covered.')
            assert not np.any(np.isnan(ret))
            return ret
        else:
            raise NotImplementedError('This value_name has not been implemented.')

class Exp:
    def __init__(self, pipeline: Pipeline, dof, dir_name, key_in_json):
        self.pipeline = pipeline
        self.dof = dof
        # self.obsnum = obsnum
        self.dir_name = dir_name
        self.key_in_json = key_in_json
    


# pipelines = {
#     # For comparing hybrid pipelines
#     'fclplan': Pipeline(True, 'fcl', None, 'Plan using FCL', ),
#     'fclopt': Pipeline(False, 'fcl', 'fcldist', 'Optim using FCL'),
#     'fclplanopt': Pipeline(True, 'fcl', 'fcldist', 'Plan+Optim using FCL'),
#     # 'diffcoplan': Pipeline(True, 'diffco', None, 'Plan using DiffCo'),
#     'diffcoopt': Pipeline(False, 'diffco', 'givengrad', 'Optim using DiffCo'),
#     'diffcoplanopt': Pipeline(True, 'diffco', 'givengrad', 'Plan+Optim using DiffCo'),
#     'fclplandiffcoopt': Pipeline(True, 'fcl', 'givengrad', 'FCL Plan+DiffCo Optim'),
# } 
pipelines = {
    'slsqpopt': Pipeline(False, 'diffco', 'givengrad', 'SLSQP'),
    'trustopt': Pipeline(False, 'diffco', 'trust-constr', 'Trust-Region Constrained'),
    'adamopt': Pipeline(False, 'diffco', 'adamdiffco', 'Adam'),
}
dofs = ['{}dof'.format(d) for d in [3]] #2, 3, 
obsnums = [1,2,5,10,20]

# exps = {
#     # For comparing hybrid pipelines
#     dof: [
#         Exp(pipelines['fclplan'], dof, f'2d_plan_test7', 'fcldist'), 
#         Exp(pipelines['fclopt'], dof, f'2d_opt_test6', 'fcldist'), 
#         Exp(pipelines['fclplanopt'], dof, f'2d_planopt_test5', 'fcldist'),
#         # Exp(pipelines['diffcoplan'], dof, f'2d_plan_test15', 'givengrad'),
#         Exp(pipelines['diffcoopt'], dof, f'2d_opt_test13', 'givengrad'),
#         Exp(pipelines['diffcoplanopt'], dof, f'2d_planopt_test14', 'givengrad'),
#         Exp(pipelines['fclplandiffcoopt'], dof, f'2d_planopt_test12', 'givengrad'),
#     ] for dof in dofs
# }
exps = {
    # For comparing hybrid pipelines
    dof: [
        Exp(pipelines['slsqpopt'], dof, '2d_plan_test17', 'givengrad'),
        Exp(pipelines['trustopt'], dof, '2d_plan_test17', 'trust-constr'),
        Exp(pipelines['adamopt'], dof, '2d_plan_test17', 'adamdiffco'),
    ] for dof in dofs
}


keys = ['success', 'time', 'cnt_check', 'cost']
# repair_keys = ['repair_'+k for k in keys]
# plan_keys = ['plan_'+k for k in keys]
# optim_keys = ['optim_'+k for k in keys]
# val_keys = ['val_time']
# sanity_check_keys = ['solution', 'repair_solution']
# all_keys = keys+repair_keys+plan_keys+optim_keys+val_keys+sanity_check_keys




# stats_by_obsnum = {k: {d: {n: {p: [] for p in pipelines}
#                            for n in obsnums} for d in dofs} for k in keys}

# result_folders = ['2d_2dof_plan_test3', '2d_3dof_plan_test3', '2d_7dof_plan_test3', '2d_2dof_plan_test4', '2d_3dof_plan_test4', '2d_7dof_plan_test4']
# for rfolder, dof in zip(result_folders, dofs):
#     for n in obsnums:
#         for fn in sorted(glob('results/'+rfolder+'/*_{}obs_*.json'.format(n))):
#             if not '{}obs'.format(n) in fn:
#                 continue
#             with open(fn, 'r') as f:
#                 r = json.load(f)
#             for m in r: 
#                 if m not in methods:
#                     continue
#                 for k in all_keys:
#                     stats_by_obsnum[k][dof][n][m].append(r[m][k])

# for k in all_keys:
#     for dof in stats_by_obsnum[k]:
#         for n in stats_by_obsnum[k][dof]:
#             for m in stats_by_obsnum[k][dof][n]:
#                 if len(stats_by_obsnum[k][dof][n][m]) == 0:
#                     print('Missing: ', k, dof, n, m)
#                     continue
#                 stats_by_obsnum[k][dof][n][m] = np.concatenate(stats_by_obsnum[k][dof][n][m])

# Sanity Check
# std_method = 'fcldist'
# for dof in stats_by_obsnum['solution']:
#     for n in stats_by_obsnum['solution'][dof]:
#         if len(stats_by_obsnum['solution'][dof][n][std_method]) == 0: # fcldist
#             continue
#         s = stats_by_obsnum['solution'][dof][n][std_method][:, [0, -1]]
#         for m in stats_by_obsnum['solution'][dof][n]:
#             assert type(stats_by_obsnum['solution'][dof][n][m]) == np.ndarray, \
#                 (m, type(stats_by_obsnum['solution'][dof][n][m]), stats_by_obsnum['solution'][dof][n][m])
#             sprime = stats_by_obsnum['repair_solution'][dof][n][m][:, [0, -1]]
#             if not (s.shape == sprime.shape and np.all(s == sprime)):
#                 print('Error::::', dof, n, m, s.shape, sprime.shape)

# print('Sanity check finished.')
# exit(1)

# plot_dof_groups = [[dof] for dof in list(stats_by_obsnum[keys[0]].keys())] # could be [dofs] to merge dofs

w = 0.5
x = np.arange(len(obsnums))*w*(len(pipelines) + 1)
fig = plt.figure(figsize=(3*len(keys), 3*len(dofs)))

keylabeltext = {
    'success': 'Success Rate',
    'time': 'Time (s)',
    'cnt_check': 'No. of Collision Checks',
    'cost': 'Cost'
}
from string import ascii_lowercase

for j, dof in enumerate(dofs):
    for i, k in enumerate(keys, 1):
        ax = fig.add_subplot(len(dofs), len(keys), j*len(keys)+i)
        ax.grid(True, 'both')
        if k == 'success':
            ax.set_ylim(0, 1)

        for itm, exp in enumerate(exps[dof]):
            cm = plt.get_cmap('Paired')

            # Only count costs of successful plans. Unsuccessful ones do not make sense.
            tmp_datas = {}
            # tmp_repair_datas = {}
            for n in obsnums:
                dlist = []
                for fn in sorted(glob('results/'+exp.dir_name+'/*_{}obs_*.json'.format(n))):
                    if '{}obs'.format(n) not in fn or dof not in fn:
                        continue
                    with open(fn, 'r') as f:
                        r = json.load(f)
                    dlist.append(exp.pipeline.get_value(k, r[exp.key_in_json]))
                if len(dlist) == 0:
                    tmp_datas[n] = dlist
                    # tmp_repair_datas[n] = rlist
                    continue
                tmp_datas[n] = np.concatenate(dlist)
                # tmp_repair_datas[n] = np.concatenate(rlist)

            mean_val = np.array([np.mean(tmp_datas[n]) for n in obsnums])
            # mean_repair_val = np.array([np.mean(tmp_repair_datas[n]) for n in obsnums])
            # if k in ['time', 'cnt_check']:
            #     mean_repair_val += mean_val
            # if all([len(stats_by_obsnum['success'][dof][n][m]) > 0 for dof in dof_group for n in stats_by_obsnum['success'][dof]]):
            if all([len(tmp_datas[n]) > 0 for n in obsnums]):
                min_val = np.array([np.min(tmp_datas[n]) for n in obsnums], dtype=np.float64)
                max_val = np.array([np.max(tmp_datas[n]) for n in obsnums], dtype=np.float64)
                # min_repair = np.array([np.min(tmp_repair_datas[n]) for n in obsnums], dtype=np.float64)
                # max_repair = np.array([np.max(tmp_repair_datas[n]) for n in obsnums], dtype=np.float64)
                # if k in ['time', 'cnt_check']:
                #     min_repair += mean_val
                #     max_repair += mean_val
            else:
                print('empty list: ', dof, exp)
                min_val = mean_val
                max_val = mean_val
                # min_repair = mean_repair_val
                # max_repair = mean_repair_val
            
            ax.bar(x+itm*w, mean_val, width=w, yerr=None if k == 'success'
                   else (mean_val-min_val, max_val-mean_val), 
                   error_kw=dict(lw=0.5, capsize=3, capthick=1.5, ecolor='darkgray'), label=exp.pipeline.labeltxt,
                   color=cm(itm*2+1),)
            # if 'fcl' not in m:
            #     ax.bar(x+itm*w, mean_repair_val-mean_val if k != 'cost' else 0, bottom=mean_val, width=w, yerr=None if True #k == 'success' #mean_repair_val if k == 'cost' else mean_val
            #         else (mean_repair_val-min_repair, max_repair-mean_repair_val), 
            #         error_kw=dict(lw=0.5, capsize=2, capthick=0.8, ecolor='darkgray'), label=methods_labeltext[m]+' (repair stage)',
            #         color=cm(itm*2),
            #         #alpha=0.7
            #         )
            

        # if len(plot_dof_groups) > 1:
        ax.set_title("({}) {} - {}DOF".format(
            ascii_lowercase[j*len(keys)+i-1],
            keylabeltext[k],
            dof.replace('dof', ''),))
        # else:
        #     ax.set_title("({}) {}".format(
        #         ascii_lowercase[j*len(keys)+i-1],
        #         keylabeltext[k],))
        ax.set_xticks([])
        ax.set_xticks(x + len(exps[dof])//2*w)
        ax.set_xticklabels(['{}obs'.format(n) for n in obsnums])
        ax.tick_params(axis='y', which='major', pad=-5)
        ax.tick_params(axis='x', which='major', pad=-4)
        if k not in ['success']:
            ax.set_yscale('log')
    if j == len(dofs)-1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                    ncol=len(pipelines), bbox_to_anchor=(0.5, -0.1), borderaxespad=1) #-0.2/len(dofs) len(exps[dof])
plt.tight_layout()
# plt.show()
# makedirs('figs/compare_planning', exist_ok=True)
# plt.savefig('figs/compare_planning/hybrid.pdf', dpi=500, bbox_inches='tight')
makedirs('figs/compare_optimizer', exist_ok=True)
plt.savefig('figs/compare_optimizer/different_optimizers.pdf', dpi=500, bbox_inches='tight')