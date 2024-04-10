import json
import os
from os.path import basename, splitext, join, isdir
from diffco import DiffCo, MultiDiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from diffco import utils
from diffco.Obstacles import FCLObstacle
from diffco.FCLChecker import FCLChecker
from time import time
from tqdm import tqdm
from motion_planner import MotionPlanner
from itertools import product
from trajectory_optim import adam_traj_optimize, givengrad_traj_optimize, gradient_free_traj_optimize, trustconstr_traj_optimize

# A function that controls the style of visualization for debugging.
def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    if robot.dof > 2: # or True: temp
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) #, projection='3d'
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)
        fig = plt.figure(figsize=(3*(num_class + 1)+0.5, 3 * num_class))
        gs = fig.add_gridspec(num_class, num_class+1)
        ax = fig.add_subplot(gs[:, :-1]) #sum([list(range(r*(num_class+1)+1, (r+1)*(num_class+1))) for r in range(num_class)], [])) #, projection='3d'
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2)).type(checker.gains.dtype)
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []
        with sns.axes_style('ticks'):
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[cat, -1])

                # score_DiffCo = checker.score(grid_points).reshape(size)
                # score = (torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.max())
                score = score_spline[:, :, cat]
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains.reshape(len(checker.gains), -1)[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                c_ax.contour(xx, yy, score, levels=[0], linewidths=1, alpha=0.4, ) #-1.5, -0.75, 0, 0.3
                # fig.colorbar(color_mesh, ax=c_ax)
                # sparse_score = score[5:-5:10, 5:-5:10]
                # score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
                # score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
                # score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
                # score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
                # score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
                # c_ax.quiver(xx[5:-5:10, 5:-5:10], yy[5:-5:10, 5:-5:10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
                # cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
                # c_ax.add_patch(cfg_point)
                cfg_path, = c_ax.plot([], [], '-o', c='orange', markersize=3)
                cfg_path_plots.append(cfg_path)

                c_ax.set_aspect('equal', adjustable='box')
                # c_ax.axis('equal')
                c_ax.set_xlim(-np.pi, np.pi)
                c_ax.set_ylim(-np.pi, np.pi)
                c_ax.set_xticks([-np.pi, 0, np.pi])
                c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
                c_ax.set_yticks([-np.pi, 0, np.pi])
                c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])
                # c_ax.tick_params(direction='in', reset=True)
                # c_ax.tick_params(which='both', direction='out', length=6, width=2, colors='r',
                #    grid_color='r', grid_alpha=0.5)
            # c_ax.set_ticks('')
    else:
        raise NotImplementedError('Unsupported degree of freedom {}'.format(robot.dof))

    # Plot ostacles
    # ax.axis('tight')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    for obs in obstacles:
        cat = obs[3] if len(obs) >= 4 else 1
        # print('{}, cat {}, {}'.format(obs[0], cat, obs))
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], color=cmaps[cat](0.5))) #path_effects=[path_effects.withSimplePatchShadow()], 
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], 
            color=cmaps[cat](0.5))) #path_effects=[path_effects.withSimplePatchShadow()], 
            # print((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    if robot.dof > 2: # or True: temp
        return fig, ax, link_plot, joint_plot, eff_plot
    elif robot.dof == 2:
        return fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots
    
def single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    from copy import copy
    from matplotlib.lines import Line2D
    points_traj = robot.fkine(p)
    points_traj = torch.cat([torch.zeros(len(p), 1, 2, dtype=points_traj.dtype), points_traj], dim=1)
    traj_alpha = 0.3
    ends_alpha = 0.5
    
    lw = link_plot.get_lw()
    link_traj = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=traj_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    # joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw/2)[0] for points in points_traj]
    eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=traj_alpha, markersize=lw/2)[0] for points in points_traj]
    # for link_plot, joint_plot, eff_plot, points in zip(link_traj, joint_traj, eff_traj, points_traj):
    #     link_plot.set_data(points[:, 0], points[:, 1])
    #     joint_plot.set_data(points[:-1, 0], points[:-1, 1])
    #     eff_plot.set_data(points[-1:, 0], points[-1:, 1])
    for i in [0, -1]:
        link_traj[i].set_alpha(ends_alpha)
        # link_traj[i].set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
        # joint_traj[i].set_alpha(ends_alpha)
        eff_traj[i].set_alpha(ends_alpha)
    link_traj[0].set_color('green')
    link_traj[-1].set_color('orange')
    # ax.add_artist(link_traj[2])

    # def divide(p): # divide the path into several segments that obeys the wrapping around rule [TODO]
    #     diff = torch.abs(p[:-1]-p[1:])
    #     div_idx = torch.where(diff.max(1) > np.pi)
    #     div_idx = torch.cat([-1, div_idx])
    #     segments = []
    #     for i in range(len(div_idx)-1):
    #         segments.append(p[div_idx[i]+1:div_idx[i+1]+1])
    #     segments.append(p[div_idx[-1]+1:])
    #     for i in range(len(segments)-1):
    #         if torch.sum(torch.abs(segments[i]) > np.pi) == 2:


    for cfg_path in cfg_path_plots:
        cfg_path.set_data(p[:, 0], p[:, 1])

    # ---------Just for making one particular figure------------
    # segments = [p[:-3], p[-3:]]
    # d1 = segments[0][-1, 0]-(-np.pi)
    # d2 = np.pi - segments[1][0, 0]
    # dh = segments[1][0, 1] - segments[0][-1, 1]
    # intery = segments[0][-1, 1] + dh/(d1+d2)*d1
    # segments[0] = torch.cat([segments[0], torch.FloatTensor([[-np.pi, intery]])])
    # segments[1] = torch.cat([torch.FloatTensor([[np.pi, intery]]), segments[1]])
    # for cfg_path in cfg_path_plots:
    #     for seg in segments:
    #         cfg_path.axes.plot(seg[:, 0], seg[:, 1], '-o', c='olivedrab', alpha=0.5, markersize=3)
    # ---------------------------------------------

    # plt.show()

class ExpConfigs(object):
    # A simple class to store experiment configurations. 
    # arguments not mentioned in args will be in their default values
    def __init__(self, args: dict):
        # default values
        self.load_exp = None
        self.include_validate_time = True
        self.use_previous_solution = False
        self.validate_density = 1
        self.only_repair = False
        self.use_planning = False
        self.use_repair = False
        self.use_optim = False
        self.valid_checker = None
        self.num_query_per_env = 10
        self.num_obs = None
        self.debug_plot_interval = 0
        self.load_key_in_json = None

        for k, v in args.items():
            assert hasattr(self, k)
            setattr(self, k, v)
        
        if self.only_repair:
            assert self.use_repair
        if isinstance(self.num_obs, int):
            self.num_obs = [self.num_obs]    

def test_one_env(env_name, optim_method, folder, args: ExpConfigs, prev_rec=None, res_folder=None):
    # assert (args.use_previous_solution and prev_rec != None) or \
    #     ((not args.use_previous_solution) and prev_rec == None), \
    #         "args.use_previous_solution does not match the existence of prev_rec."
    print(env_name, optim_method, 'Begin')
    # Prepare distance estimator ====================
    dataset = torch.load('{}/{}.pt'.format(folder, env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].double() #.max(1).values
    dists = dataset['dist'].double() #.reshape(-1, 1) #.max(1).values
    obstacles = dataset['obs']
    # obstacles = [obs+(i, ) for i, obs in enumerate(obstacles)]
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine

    train_t = time()
    checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    if not args.use_previous_solution: # or robot.dof == 2 or args.use_optim:
        checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

        # Check DiffCo test ACC
        test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
        test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
        test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
        test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
        print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
        if test_acc < 0.9:
            print('test acc is only {}'.format(test_acc))

        fitting_target = 'label' # {label, dist, hypo}
        Epsilon = 1 #0.01
        checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target)
        # checker.fit_full_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine)#, lmbd=80)
        # ========================
        # ONLY for additional training timing exp
        # fcl_obs = [FCLObstacle(*param) for param in obstacles]
        # fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
        # obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        # obs_managers[0].registerObjects(fcl_collision_obj)
        # obs_managers[0].setup()
        # robot_links = robot.update_polygons(cfgs[0])
        # robot_manager = fcl.DynamicAABBTreeCollisionManager()
        # robot_manager.registerObjects(robot_links)
        # robot_manager.setup()
        # for mng in obs_managers:
        #     mng.setup()
        # gt_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)
        # gt_checker.predict(cfgs[:train_num], distance=False)
        # return time() - train_t
        # END ========================
        dist_est = checker.rbf_score
        # dist_est = checker.poly_score
        min_score = dist_est(cfgs[train_num:]).min().item()
        # print('MIN_SCORE = {:.6f}'.format(min_score))
    else:
        dist_est = None
        min_score = None
    # ==============================================

    # FCL checker =====================
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

    label_type = 'binary'
    num_class = 1

    if label_type == 'binary':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        obs_managers[0].registerObjects(fcl_collision_obj)
        obs_managers[0].setup()
    elif label_type == 'instance':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in fcl_obs]
        for mng, cobj in zip(obs_managers, fcl_collision_obj):
            mng.registerObjects([cobj])
    elif label_type == 'class':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in range(num_class)]
        obj_by_cls = [[] for _ in range(num_class)]
        for obj in fcl_obs:
            obj_by_cls[obj.category].append(obj.cobj)
        for mng, obj_group in zip(obs_managers, obj_by_cls):
            mng.registerObjects(obj_group)
    else:
        raise NotImplementedError('Unsupported label_type {}'.format(label_type))
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    for mng in obs_managers:
        mng.setup()
    fcl_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)
    # =================================

    optim_options = {
        'N_WAYPOINTS': 20,
        'NUM_RE_TRIALS': 3,
        'MAXITER': 200,
        'safety_margin': 1/3*min_score if min_score is not None else None, 
        'max_speed': 1.5,
        'seed': 1234,
        'history': False
    }

    repair_options = {
        'N_WAYPOINTS': 20,
        'NUM_RE_TRIALS': 1, # just one trial
        'MAXITER': 200,
        'max_speed': 1.5,
        'seed': 1234, # actually not used due to only one trial
        'history': False,
    }

    planning_options = {
        'maxtime': 30
    }

    # test_rec = {
    #     'start_cfg': [],
    #     'target_cfg': [],
    #     'cnt_check': [],
    #     'repair_cnt_check': [],
    #     'cost': [],
    #     'repair_cost': [],
    #     'time': [],
    #     'val_time': [],
    #     'repair_time': [],
    #     'success': [],
    #     'repair_success': [],
    #     'seed': [],
    #     'solution': [],
    #     'repair_solution': [],
    # }
    test_rec = {}

    if args.use_planning:
        if args.valid_checker is None:
            raise ValueError(f'Planning is requested but no validity checker is provided: valid_checker={args.valid_checker}.')
        elif args.valid_checker.lower() == 'fcl':
            checker_func = lambda cfg: fcl_checker(cfg)[0].item() < 0
        elif args.valid_checker.lower() == 'diffco':
            checker_func = lambda cfg: dist_est(cfg)[0].item() - optim_options['safety_margin'] < 0
        else:
            raise NotImplementedError
        def motion_cost_func(s1, s2):
            p_tensor = torch.stack([s1, s2])
            control_points = robot.fkine(p_tensor)
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            return diff.item()
        mp = MotionPlanner(robot, checker_func, motion_cost_func)
    
    with open('{}/{}_testcfgs.json'.format(folder, env_name), 'r') as f:
        test_cfg_dataset = json.load(f)
        s_cfgs = torch.FloatTensor(test_cfg_dataset['start_cfgs'])[:args.num_query_per_env]
        t_cfgs = torch.FloatTensor(test_cfg_dataset['target_cfgs'])[:args.num_query_per_env]
        assert env_name == test_cfg_dataset['env_name']
    # if prev_rec != None:
    #     rec_s_cfgs = torch.FloatTensor([sol[0] for sol in prev_rec['solution']])[:args.num_query_per_env]
    #     rec_t_cfgs = torch.FloatTensor([sol[-1] for sol in prev_rec['solution']])[:args.num_query_per_env] # torch.FloatTensor(prev_rec['solution'])[:, -1]
    #     assert torch.all(torch.isclose(rec_s_cfgs, s_cfgs[:len(rec_s_cfgs)])) and torch.all(torch.isclose(rec_t_cfgs, t_cfgs[:len(rec_t_cfgs)]))
    
    
    for test_it, (start_cfg, target_cfg) in tqdm(enumerate(zip(s_cfgs, t_cfgs)), total=len(s_cfgs), desc='Test Query'):
        optim_options['seed'] += 1 # Otherwise the random initialization will stay the same every problem

        if prev_rec != None and test_it < len(prev_rec['success']):
            print('using saved solution {}'.format(test_it))
            # tmp_rec = {k: prev_rec[k][test_it] for k in prev_rec} # this is usual case
            tmp_rec = {k: prev_rec[k][test_it] for k in prev_rec if 'repair_' not in k} # this is temporary to ignore repair stuff
        else:
            tmp_rec = {}
        
        # if (env_name, test_it) not in [ # Temp
        #     ('2d_2dof_2obs_binary_02', 9),
        #     ('2d_2dof_20obs_binary_08', 0),
        #     ('2d_3dof_10obs_binary_06', 3),
        #     ('2d_7dof_20obs_binary_03', 6),
        # ]: continue

        tmp_optim_options = optim_options # default to no planning solution
        if args.use_planning:
            if 'plan_solution' not in tmp_rec:
                plan_rec = mp.plan(start_cfg, target_cfg, planning_options)
                if plan_rec['solution'] != None and len(plan_rec['solution']) > optim_options['N_WAYPOINTS']:
                    tmp_path = torch.DoubleTensor(plan_rec['solution'])
                    original_cost = sum([motion_cost_func(tmp_path[i], tmp_path[i+1]) for i in range(len(tmp_path)-1)])
                    print('Original path cost = ', original_cost, plan_rec['cost'])
                    indices = torch.linspace(0, len(tmp_path)-1, optim_options['N_WAYPOINTS'], dtype=int)
                    tmp_path = tmp_path[indices]
                    plan_rec['cost'] = sum([motion_cost_func(tmp_path[i], tmp_path[i+1]) for i in range(len(tmp_path)-1)])
                    plan_rec['solution'] = tmp_path.numpy().tolist()
                    print('Path cost modified to ', plan_rec['cost'])
                for k in plan_rec:
                    tmp_rec['plan_'+k] = plan_rec[k]
                    tmp_rec[k] = plan_rec[k]
            if tmp_rec['plan_success']:
                tmp_optim_options = {**optim_options, 'init_solution': torch.DoubleTensor(tmp_rec['plan_solution'])}

        if args.use_optim:
            if args.use_optim == 'force' or 'optim_solution' not in tmp_rec or ('optim_method' in tmp_rec and tmp_rec['optim_method'] != optim_method):
                if optim_method == 'fclgradfree':
                    print('solving query {} with fclgradfree'.format(test_it))
                    optim_rec = gradient_free_traj_optimize(robot, lambda cfg: fcl_checker.predict(cfg, distance=False), start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'fcldist':
                    optim_rec = gradient_free_traj_optimize(robot, fcl_checker.score, start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'adamdiffco':
                    optim_rec = adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'bidiffco':
                    optim_rec = gradient_free_traj_optimize(robot, lambda cfg: 2*(dist_est(cfg)>=0).type(torch.FloatTensor)-1, start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'diffcogradfree':
                    with torch.no_grad():
                        optim_rec = gradient_free_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'margindiffcogradfree':
                    with torch.no_grad():
                        optim_rec = gradient_free_traj_optimize(robot, lambda cfg: dist_est(cfg)-tmp_optim_options['safety_margin'], start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'givengrad':
                    optim_rec = givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=tmp_optim_options)
                elif optim_method == 'trust-constr':
                    optim_rec = trustconstr_traj_optimize(robot, dist_est, start_cfg, target_cfg, options=tmp_optim_options)
                else:
                    raise NotImplementedError('Method = {} not implemented'.format(optim_method))
                
                for k in optim_rec:
                    tmp_rec['optim_'+k] = optim_rec[k]
                    tmp_rec[k] = optim_rec[k]
                tmp_rec['optim_method'] = optim_method
        
        ### ============= Verification ======================
        # if tmp_rec['success']:
        def con_max_move(p):
            control_points = robot.fkine(p)
            return torch.all((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-optim_options['max_speed']**2 <= 0).item()
        def con_collision_free(p):
            return torch.all(fcl_checker.predict(p, distance=False) < 0).item()
        def con_dist_collision_free(p):
            return torch.all(fcl_checker.score(p) < 0).item()
            # return True
        def con_joint_limit(p):
            return (torch.all(robot.limits[:, 0]-p <= 0) and torch.all(p-robot.limits[:, 1] <= 0)).item()

        def validate(solution, method=None):
            if solution is None:
                return False
            veri_cfgs = [utils.anglin(q1, q2, args.validate_density, endpoint=False)\
                for q1, q2 in zip(solution[:-1], solution[1:])]
            veri_cfgs = torch.cat(veri_cfgs, 0)
            collision_free = con_collision_free(veri_cfgs) if method != 'fcldist' else con_dist_collision_free(veri_cfgs)
            sol_tensor = torch.FloatTensor(solution)
            within_jointlimit = con_joint_limit(sol_tensor)
            within_movelimit = con_max_move(sol_tensor)
            return collision_free and within_jointlimit and within_movelimit
        
        if args.use_planning and not args.use_optim:
            solution_to_validate = tmp_rec['plan_solution']
            tmp_success = tmp_rec['plan_success']
            need_validate = 'fcl' not in args.valid_checker.lower() or args.validate_density > 1
            print('Validating a planning solution...')
        elif args.use_optim:
            solution_to_validate = tmp_rec['optim_solution']
            tmp_success = tmp_rec['optim_success']
            need_validate = 'fcl' not in optim_method.lower() or args.validate_density > 1
            print('Validating a optimization solution...')
        if not need_validate: # skip validation if using fcl and density is only 1
            val_t = 0
            # tmp_rec['success'] = validate(tmp_rec['solution'], method=method) # This is only temporary
        else:
            val_t = time()
            tmp_success = validate(solution_to_validate)
            val_t = time() - val_t
        tmp_rec['val_time'] = val_t
        tmp_rec['success'] = tmp_success
        
        ### =============== Repair ========================
        if args.use_repair:
            if not tmp_rec['success'] and 'fcl' not in optim_method:
                repair_rec = gradient_free_traj_optimize(robot, fcl_checker.score, start_cfg, target_cfg, 
                    options={**repair_options, 'init_solution': torch.DoubleTensor(tmp_rec['solution'])})
                # repair_rec['success'] = validate(repair_rec['solution']) # validation not needed
            else:
                repair_rec = {
                    'cnt_check': 0,
                    'cost': tmp_rec['cost'],
                    'time': 0,
                    'success': tmp_rec['success'],
                    'solution': tmp_rec['solution'],
                }
            if args.validate_density > 1:
                val_t = time()
                repair_rec['success'] = validate(repair_rec['solution'])
                val_t = time() - val_t
                tmp_rec['val_time'] += val_t
            for k in repair_rec:
                tmp_rec['repair_'+k] = repair_rec[k]
                tmp_rec[k] = repair_rec[k]

        # for k in tmp_rec:
        #     if k not in test_rec:
        #         test_rec[k] = []
        #     test_rec[k].append(tmp_rec[k])
        #     assert len(test_rec[k]) == test_it+1
        for k in set(list(tmp_rec.keys())+list(test_rec.keys())):
            if k not in test_rec:
                test_rec[k] = [None] * test_it
            if k in tmp_rec:
                test_rec[k].append(tmp_rec[k])
            else:
                test_rec[k].append(None)
            assert len(test_rec[k]) == test_it+1
            
        
        # ================Visualize for debugging purposes===================
        if args.debug_plot_interval > 0 and test_it % args.debug_plot_interval == 0:
            cfg_path_plots = []
            if robot.dof > 2:# or True: # TEMP
                fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker)
            elif robot.dof == 2:
                fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
            if test_rec['solution'][-1] != None:
                single_plot(robot, torch.FloatTensor(test_rec['solution'][test_it][::4] if robot.dof == 7 else test_rec['solution'][test_it]), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
            else:
                single_plot(robot, torch.stack([start_cfg, target_cfg]), fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
            if res_folder is not None:
                debug_dir = join(res_folder, 'debug', env_name, optim_method)
            else:
                debug_dir = join('debug', env_name, optim_method)
            if not isdir(debug_dir):
                os.makedirs(debug_dir)
            plt.title('Success {}, Cost {:.3f}'.format(tmp_rec['success'], tmp_rec['cost']))
            plt.savefig(join(debug_dir, 'debug_view_{}.png'.format(test_it)), dpi=500)
            plt.close()
            # break # debugging

    return test_rec

def main(optim_method, exp_name, res_folder=None, override=False, args: ExpConfigs=None):
    if args.load_exp is not None:
        print('Loading experiment results from {}'.format(args.load_exp))
    data_folder = join('data', exp_name) # if args.load_exp is None else args.load_exp)
    res_folder = join('results', exp_name if res_folder is None else res_folder)
    restored_res_folder = res_folder if args.load_exp is None else join('results', args.load_exp)
    load_key_in_json = args.load_key_in_json if args.load_key_in_json is not None else optim_method
    if not isdir(res_folder):
        os.makedirs(res_folder)
    elif not override:
        ans = input('Overriding {}. Continue?(Y/n)'.format(res_folder))
        if 'y' in ans or 'Y' in ans:
            pass
        else:
            exit(1)
    
    with open(join(res_folder, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)

    from glob import glob
    envs = sorted(glob(join(data_folder, '*.pt'),))

    for env_name in tqdm(envs):
        env_name = splitext(basename(env_name))[0]
        # if env_name not in ['2d_2dof_2obs_binary_02', '2d_3dof_10obs_binary_06', '2d_7dof_20obs_binary_03', '2d_2dof_20obs_binary_08']: # Temp
        #     continue
        if args.num_obs != None and not any([f'{n}obs' in env_name for n in args.num_obs]):
            print(f'WARNING: Skipping {env_name} because it does not have the required number of obstacles: {args.num_obs}')
            continue # Skipping the environments not containing the requested number of obs.

        # Load previously stored results. This is to enable experiment resuming.
        restore_rec_file = os.path.join(restored_res_folder, env_name+'.json')
        if os.path.isfile(restore_rec_file):
            with open(restore_rec_file, 'r') as f:
                env_all_rec = json.load(f)
            if optim_method in env_all_rec and args.load_exp is None:
                print(f'WARNING: Skipping {env_name}, {optim_method} because it\'s done.')
                continue
            elif args.only_repair and optim_method in env_all_rec and 'repair_success' in env_all_rec[optim_method]:
                assert optim_method in env_all_rec and env_all_rec[optim_method] != {}
                continue
        else: 
            assert args.load_exp is None, \
                'Trying to load experiment {}, but the result file {} does not exist'.format(args.load_exp, restore_rec_file)
            env_all_rec = {}
        
        # if all_rec != {}:
        #     print('Env {} missing Method {}, redoing now...'.format(env_name, method))
        #     # continue
        # if method == 'fcldist' and '10obs' not in env_name:
        #     # This is just temporary!!
        #     print('Skipping {}, {}'.format(env_name, method))
        #     continue
        test_rec = test_one_env(env_name, optim_method=optim_method, folder=data_folder, args=args, 
            prev_rec=env_all_rec[load_key_in_json] if args.load_exp != None else None, res_folder=res_folder)
        
        rec_file = os.path.join(res_folder, env_name+'.json')
        env_all_rec[optim_method] = test_rec
        with open(rec_file, 'w') as f:
            json.dump(env_all_rec, f, indent=4)
            print(f'WARNING: Records written to {rec_file}')

def additional_timing(method, exp_name):
    folder = join('data', exp_name)

    from glob import glob
    envs = sorted(glob(join(folder, '*.pt'),))

    train_ts = {}
    for obsn in [1,2,5,10,20]:
        train_ts[obsn] = []
        for env_name in tqdm(envs):
            if not '_{}obs_'.format(obsn) in env_name:
                continue
            env_name = splitext(basename(env_name))[0]
            t = test_one_env(env_name, optim_method=method, folder=folder)
            train_ts[obsn].append(t)
    print('{}, {}:'.format(m, exp_name))
    for obsn in [1,2,5,10,20]:
        ts = np.array(train_ts[obsn])
        print('{} train times: {} mean {} std {} '.format(obsn, ts, ts.mean(), ts.std()))
    return train_ts

if __name__ == "__main__":
    # the names of the data and the result directories
    # Both should be the sub-directory names under 'data', 'result', respectively
    env_and_queries = ['2d_3dof_exp4'] # '2d_7dof_exp4', '2d_3dof_exp4', '2d_2dof_exp4', 
    res_folders = ['2d_plan_test17',] # '2d_7dof_plan_test1', '2d_3dof_plan_test1', '2d_2dof_plan_test1', 

    # load_exps contain the names of the result directories you want to re-compute.
    # set to a list of None's when running new experiments
    # load_exps = ['2d_plan_test16']*3 # ['2d_plan_test7',] # [None] # ['2d_planopt_test5'] #, None, None]
    load_exps = [None]*3


    methods = ['adamdiffco',] # ['trust-constr', 'fclgradfree', 'fcldist', 'margindiffcogradfree', 'adamdiffco', 'bidiffco', 'diffcogradfree']
    valid_checkers = [None] # ['diffco', 'fcl'] # valid checker for the planner
    res = {}
    for (eq, res_folder, loadexp), (m, checker) in tqdm(list(product(
        zip(env_and_queries, res_folders, load_exps),
        zip(methods, valid_checkers)
    )),  desc='Experiments progress'):
        assert loadexp == None or loadexp == res_folder # so you better make a copy of an existing result folder and modify it, if you want to load any experiment
        res[eq] = {}
        # for m in methods:
        st = time()
        args = dict(
            load_exp=loadexp,
            include_validate_time=True,
            use_previous_solution=False, # Set this to False unless you want to load previous solutions
            validate_density=1, #1 means only check on waypoints. >=2 means also check some intermediate points between waypoints
            only_repair=False, # This is only to add repair for several previous experiments. Set to False
            use_planning=False, 
            use_repair=False,
            use_optim=True, # 'force' meaning rewrite the saved solution
            valid_checker=checker, # validity checker for planning, not for verification; verification always uses FCL
            num_query_per_env=2, 
            num_obs=None, 
            debug_plot_interval=3, # 0 means no debugging plots
            # load_key_in_json='fcldist'
        )
        args=ExpConfigs(args)
        main(m, eq, res_folder=res_folder, override=True, args=args) # this is the main experiment. Set override=True when filling results in the same directory
        # # res[exp_name][m] = additional_timing(m, exp_name) # This is for additional timing on initial training
        et = time()
        print('Method {}, Exp {}, time = {:.3f} secs'.format(m, eq, et-st))

    # This is to print out additional timings
    # for exp_name in exps:
    #     for m in methods:
    #         print('{}, {}:'.format(m, exp_name))
    #         for obsn in [1,2,5,10,20]:
    #             ts = np.array(res[exp_name][m][obsn])
    #             print('{} train times: {} mean {} std {} '.format(obsn, ts, ts.mean(), ts.std()))