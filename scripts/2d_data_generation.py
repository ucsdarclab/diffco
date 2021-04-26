import sys
from diffco import DiffCo
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
from time import time


if __name__ == "__main__":
    # ========================== Data generqation =========================
    env_name = '1rect_1circle_7d' # 
    label_type = 'binary' #[instance, class, binary]
    num_class = 2
    DOF = 7

    obstacles = {
        # ('circle', (3, 2), 2), #2circle
        # ('circle', (-2, 3), 0.5), #2circle
        # ('rect', (-2, 3), (1, 1)),
        # ('rect', (1.7, 3), (2, 3)),
        # ('rect', (-1.7, 3), (2, 3)),
        # ('rect', (0, -1), (10, 1)),
        # ('rect', (8, 7), 1),
        '1rect_1circle': [('rect', (4, 3), (2, 2)),
            ('circle', (-4, -3), 1)],
        # ('rect', (4, 3), (2, 2)), # 2rect
        # ('rect', (-4, -3), (2, 2)) # 2rect
        # ('rect', (3, 2), (2, 2)) # 1rect
        '3circle': [
            ('circle', (0, 4.5), 1), #3circle
            ('circle', (-2, -3), 2), #3circle
            ('circle', (-2, 2), 1.5), #3circle
        ],
        '1rect_1circle_7d': [
            ('circle', (-2, 3), 1), #1rect_1circle_7d
            ('rect', (3, 2), (2, 2)) #1rect_1circle_7d
        ],
        # ('rect', (5, 0), (2, 2), 0), #2class_1
        # ('circle', (-3, 6), 1, 1), #2class_1
        # ('rect', (-5, 2), (2, 1.5), 1), #2class_1
        # ('circle', (-5, -2), 1.5, 1), #2class_1 
        # ('circle', (-3, -6), 1, 1) #2class_1
        # ('rect', (0, 3), (16, 0.5), 1), #2class_2
        # ('rect', (0, -3), (16, 0.5), 0), #2class_2
        # ('rect', (-7, 3), (2, 2)) #1rect_active
        '3circle_7d': [
            ('circle', (-2, 2), 1), #3circle_7d
            ('circle', (-3, 3), 1), #3circle
            ('circle', (-6, -3), 1) #3circle
        ]
        # ('rect', (5, 4), (4, 4), 0), #2instance_big
        # ('circle', (-5, -4), 2, 1) #2instance_big
    }
    obstacles = obstacles[env_name]
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    # geom2instnum = {id(g): i for i, (_, g) in enumerate(fcl_obs)}
    
    width = 0.3
    link_length = 1
    robot = RevolutePlanarRobot(link_length, width, DOF) # (7, 1), (2, 3)

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
    if 'compare' not in env_name or DOF > 2:
        cfgs = 2*(torch.rand((num_init_points, DOF), dtype=torch.float32)-0.5) * np.pi
    else:
        # --- only for compare with gt distance
        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        cfgs = torch.stack([xx, yy], axis=2).reshape((-1, DOF))
        num_init_points = len(cfgs)
        # --- only for compare with gt distance
    if label_type == 'binary':
        labels = torch.zeros(num_init_points, 1, dtype=torch.float)
        dists = torch.zeros(num_init_points, 1, dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        obs_managers[0].registerObjects(fcl_collision_obj)
        obs_managers[0].setup()
    elif label_type == 'instance':
        labels = torch.zeros(num_init_points, len(obstacles), dtype=torch.float)
        dists = torch.zeros(num_init_points, len(obstacles), dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in fcl_obs]
        for mng, cobj in zip(obs_managers, fcl_collision_obj):
            mng.registerObjects([cobj])
    elif label_type == 'class':
        labels = torch.zeros(num_init_points, num_class, dtype=torch.float)
        dists = torch.zeros(num_init_points, num_class, dtype=torch.float)
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in range(num_class)]
        obj_by_cls = [[] for _ in range(num_class)]
        for obj in fcl_obs:
            obj_by_cls[obj.category].append(obj.cobj)
        for mng, obj_group in zip(obs_managers, obj_by_cls):
            mng.registerObjects(obj_group)
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    for mng in obs_managers:
        mng.setup()
    req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
    
    times = []
    st = time()
    for i, cfg in enumerate(cfgs):
        st1 = time()
        robot.update_polygons(cfg)
        robot_manager.update()
        assert len(robot_manager.getObjects()) == DOF
        for cat, obs_mng in enumerate(obs_managers):
            rdata = fcl.CollisionData(request = req)
            robot_manager.collide(obs_mng, rdata, fcl.defaultCollisionCallback)
            in_collision = rdata.result.is_collision
            ddata = fcl.DistanceData()
            robot_manager.distance(obs_mng, ddata, fcl.defaultDistanceCallback)
            depths = torch.FloatTensor([c.penetration_depth for c in rdata.result.contacts])

            labels[i, cat] = 1 if in_collision else -1
            dists[i, cat] = depths.abs().max() if in_collision else -ddata.result.min_distance
        end1 = time()
        times.append(end1-st1)
    end = time()
    times = np.array(times)
    print('std: {}, mean {}, avg {}'.format(times.std(), times.mean(), (end-st)/len(cfgs)))
    
    in_collision = (labels == 1).sum(1) > 0
    if label_type == 'binary':
        labels = labels.squeeze_(1)
        dists = dists.squeeze_(1)

    

    print('{} collisons, {} free'.format(torch.sum(in_collision==1), torch.sum(in_collision==0)))
    dataset = {'data': cfgs, 'label': labels, 'dist': dists, 'obs': obstacles, 'robot': robot.__class__, 'rparam': [link_length, width, DOF, ]}
    torch.save(dataset, 'data/2d_{}dof_{}.pt'.format(DOF, env_name))
    # ========================== Data generqation =========================

    # DOF = 2
    # env_name = '1rect'

    # dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    # cfgs = dataset['data']
    # labels = dataset['label']
    # dists = dataset['dist']
    # obstacles = dataset['obs']
    # robot = dataset['robot'](*dataset['rparam'])
    # width = robot.link_width
    # train_num = 3000
    # fkine = robot.fkine
    # checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # # Check DiffCo test ACC
    # test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    # test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds)
    # test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    # test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    # print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    # assert(test_acc > 0.9)

    # fitting_target = 'label' # {label, dist, hypo}
    # Epsilon = 0.01
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine)
    # # checker.fit_full_poly(k=1, epsilon=Epsilon, target=fitting_target, fkine=fkine)
    # spline_func = checker.rbf_score
    # # spline_func = checker.poly_score

    # collision_cfgs = cfgs[labels==1]
    # # collision_cfgs = cfgs

    # if DOF == 7:
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(111) #, projection='3d'
    # elif DOF == 2:
    #     # Show C-space at the same time
    #     fig = plt.figure(figsize=(16, 8))
    #     ax = fig.add_subplot(121) #, projection='3d'
    #     c_ax = fig.add_subplot(122)
    #     size = [400, 400]
    #     yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    #     grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    #     score_spline = spline_func(grid_points).reshape(size)
    #     score_DiffCo = checker.score(grid_points).reshape(size)
    #     score = (torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_DiffCo)+1)/2*(score_spline-score_spline.max())
    #     # score = score_spline
    #     c = c_ax.pcolormesh(xx, yy, score, cmap='RdBu_r', vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
    #     c_ax.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
    #     c_ax.contour(xx, yy, score, levels=0)
    #     c_ax.axis('equal')
    #     fig.colorbar(c, ax=c_ax)
    #     sparse_score = score[::10, ::10]
    #     score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
    #     score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
    #     score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    #     score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    #     score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    #     c_ax.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
    #     cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
    #     c_ax.add_patch(cfg_point)


    # ax.axis('equal')
    # ax.set_xlim(-8, 7)
    # ax.set_ylim(-8, 7)
    # ax.set_aspect('equal', adjustable='box')
    # for obs in obstacles:
    #     if obs[0] == 'circle':
    #         ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()]))
    #     elif obs[0] == 'rect':
    #         ax.add_patch(Rectangle((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()]))
    

    # trans = ax.transData.transform
    # lw = ((trans((1, width))-trans((0,0)))*72/ax.figure.dpi)[1]
    # link_plot, = ax.plot([], [], color='silver', lw=lw, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()], solid_capstyle='round')
    # joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    # eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    # global cur_cfg, cfg_cnt, opt, start_frame, cnt_down
    # lr = 5e-2
    # decay_weight = 1.0 # 0.1
    # decay_loss = torch.nn.MSELoss() #reduction='sum'
    # FPS = 15
    # pause_t = 0.5 # seconds
    # grad_clip = False
    # optimizer = torch.optim.Adam
    # def init():
    #     global cur_cfg, cfg_cnt, opt, start_frame, start_cfg, cnt_down
    #     cur_cfg = collision_cfgs[0].clone().detach().requires_grad_(True)
    #     start_cfg = collision_cfgs[0].clone().detach().requires_grad_(False)
    #     cfg_cnt = 0
    #     start_frame = 0
    #     cnt_down = int(pause_t*FPS)
    #     opt = optimizer([cur_cfg], lr=lr)
    #     if DOF==2:
    #         return link_plot, joint_plot, eff_plot, cfg_point
    #     else:
    #         return link_plot, joint_plot, eff_plot
    
    # def plot_robot(q):
    #     global cur_cfg, cfg_cnt, opt
    #     robot_points = robot.fkine(cur_cfg)[0]
    #     robot_points = torch.cat([torch.zeros(1, 2), robot_points])
    #     link_plot.set_data(robot_points[:, 0], robot_points[:, 1])
    #     joint_plot.set_data(robot_points[:-1, 0], robot_points[:-1, 1])
    #     eff_plot.set_data(robot_points[-1:, 0], robot_points[-1:, 1])

    #     return link_plot, joint_plot, eff_plot


    # def update(i):
    #     global cur_cfg, cfg_cnt, opt, start_frame, start_cfg, cnt_down
    #     with torch.no_grad():
    #         ret = plot_robot(cur_cfg)
    #         in_collision = checker.is_collision(cur_cfg)
    #     if DOF == 2:
    #         cfg_point.set_center(cur_cfg)
    #     opt.zero_grad()
    #     score = spline_func(cur_cfg)
    #     movement_loss = decay_weight * decay_loss(fkine(cur_cfg), fkine(start_cfg))
    #     loss = score + movement_loss
    #     loss.backward()
        
    #     print('CFG {} Frame {}: Score {:.8f}, movement_loss {:.8f}, grad {:.8f}'.format(
    #         cfg_cnt, i, score.data.numpy(), movement_loss.data.numpy(), cur_cfg.grad.norm().numpy()))
    #     # print(cur_cfg.grad.numpy())
    #     if  (score < -0.5 and not in_collision) or \
    #         (cur_cfg.grad.norm() < 10 and (i-start_frame) > 200):
    #         ax.set_title(('grad norm: {:.1f}, in collision: {}, score: {:.1f}, i - start_frame: {}'.format(
    #             cur_cfg.grad.norm(), 
    #             in_collision, 
    #             score,
    #             i-start_frame)))
    #         if not cnt_down:
    #             cfg_cnt += 1
    #             start_frame = i
    #             cur_cfg = collision_cfgs[cfg_cnt].clone().detach().requires_grad_(True)
    #             start_cfg = collision_cfgs[cfg_cnt].clone().detach().requires_grad_(False)
    #             opt = optimizer([cur_cfg], lr=lr)
    #             cnt_down = int(pause_t * FPS)
    #         else:
    #             cnt_down -= 1
    #     else:
    #         if grad_clip:
    #             # torch.nn.utils.clip_grad(cur_cfg, 20)
    #             cur_cfg.grad.clamp_(-2, 2)
    #         opt.step()
    #         cur_cfg.data = utils.wrap2pi(cur_cfg.data)
    #         ax.set_title('Configuration {}, Score {:.1f}, Collision {}'.format(cfg_cnt, score.item(), in_collision))
    #     if DOF == 2:
    #         return ret+(cfg_point, )
    #     else:
    #         return ret
    
    # ani = animation.FuncAnimation(fig, update, frames=900, interval=1, blit=False, init_func=init)
    # plt.show()
    # ani.save('results/2d_{}dof_{}_gradclip_{}_fitting_{}_decay_{}_eps_{}.mp4'.format(
    #     DOF, env_name, grad_clip, fitting_target, decay_weight, Epsilon), fps=FPS)


