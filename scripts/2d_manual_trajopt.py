import sys
sys.path.append('/home/yuheng/FastronPlus-pytorch/')
from Fastronpp import Fastron
from Fastronpp import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from Fastronpp.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from Fastronpp import utils
from Fastronpp.Obstacles import FCLObstacle


if __name__ == "__main__":
    DOF = 7
    env_name = '1rect_1circle' # '2rect' # '1rect_1circle' '1rect' 'narrow'

    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data']
    labels = dataset['label']
    dists = dataset['dist']
    obstacles = dataset['obs']
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width
    train_num = 6000
    fkine = robot.fkine
    checker = Fastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # Check Fastron test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds)
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.9)

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine) # epsilon=Epsilon,
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine)
    spline_func = checker.rbf_score
    # spline_func = checker.poly_score


    if DOF == 7:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111) #, projection='3d'
    elif DOF == 2:
        # Show C-space at the same time
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121) #, projection='3d'
        c_ax = fig.add_subplot(122)
        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
        score_spline = spline_func(grid_points).reshape(size)
        score_fastron = checker.score(grid_points).reshape(size)
        score = (torch.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
        # score = score_spline
        c = c_ax.pcolormesh(xx, yy, score, cmap='RdBu_r', vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
        c_ax.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
        c_ax.contour(xx, yy, score, levels=0)
        c_ax.axis('equal')
        fig.colorbar(c, ax=c_ax)
        sparse_score = score[::10, ::10]
        score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
        score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
        score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
        score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
        score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
        c_ax.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
        # cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
        # c_ax.add_patch(cfg_point)
        cfg_path, = c_ax.plot([], [], '-o', c='orange')

        # cfg_waypoints, = c_ax.plot([], [], 'o', c='silver')

    # Plot ostacles
    ax.axis('equal')
    ax.set_xlim(-8, 7)
    ax.set_ylim(-8, 7)
    ax.set_aspect('equal', adjustable='box')
    for obs in obstacles:
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()]))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()]))
            print((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot([], [], color='silver', lw=lw, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()], solid_capstyle='round')
    joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    # Begin optimization
    torch.manual_seed(1988)
    free_cfgs = cfgs[labels == -1]
    indices = torch.randint(0, len(free_cfgs), (2, ))
    while indices[0] == indices[1]:
        indices = torch.randint(0, len(free_cfgs), (2, ))
    start_cfg = torch.zeros(DOF, dtype=torch.float32) # free_cfgs[indices[0]] # 
    target_cfg = torch.zeros(DOF, dtype=torch.float32) # free_cfgs[indices[1]] # 
    target_cfg[0] = np.pi/2
    N_STEPS = 20
    # p = torch.FloatTensor(np.concatenate([np.linspace(start_cfg, (-np.pi, 0), N_STEPS/2), np.linspace((np.pi, 0), target_cfg, N_STEPS/2)], axis=0)).requires_grad_(True)
    p = (torch.rand(N_STEPS, DOF))*np.pi*2-np.pi
    p[0] = start_cfg
    p[-1] = target_cfg
    p.requires_grad_(True)

    global cur_cfg, cfg_cnt, opt, start_frame, cnt_down
    lr = 5e-1
    decay_loss = torch.nn.MSELoss() #reduction='sum'
    FPS = 15
    pause_t = 0.5 # seconds
    # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)
    opt = torch.optim.Adam([p], lr=lr)
    def init():
        if DOF==2:
            return link_plot, joint_plot, eff_plot, cfg_path
        else:
            return link_plot, joint_plot, eff_plot

    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 10
    collision_weight = 10
    def update_traj(i):
        opt.zero_grad()
        score = collision_weight * torch.clamp(spline_func(p), min=-5).sum()
        control_points = fkine(p)
        max_move_cost = max_move_weight * torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2), min=0.3**2).sum()
        diff = dif_weight * (control_points[1:]-control_points[:-1]).pow(2).sum()
        # np.clip(1.5*float(i)/UPDATE_STEPS, 0, 1)**2 (float(i)/UPDATE_STEPS) * 
        # torch.clamp(utils.wrap2pi(p[1:]-p[:-1]).abs(), min=0.3).pow(2).sum()
        loss = score + diff + max_move_cost
        loss.backward()
        p.grad[[0, -1]] = 0.0
        opt.step()
        p.data = utils.wrap2pi(p.data)
        if DOF == 2:
            cfg_path.set_data(p.data[:, 0], p.data[:, 1])
            return cfg_path,
        else:
            return link_plot, joint_plot, eff_plot
        

    def plot_robot(q):
        robot_points = robot.fkine(q)[0]
        robot_points = torch.cat([torch.zeros(1, 2), robot_points])
        link_plot.set_data(robot_points[:, 0], robot_points[:, 1])
        joint_plot.set_data(robot_points[:-1, 0], robot_points[:-1, 1])
        eff_plot.set_data(robot_points[-1:, 0], robot_points[-1:, 1])

        return link_plot, joint_plot, eff_plot

    def animate_robot(i):
        i = i if i < len(p) else len(p)-1
        with torch.no_grad():
            ret = plot_robot(p[i])
        # if DOF == 2:
        #     cfg_point.set_center(cur_cfg)
        return ret
        
  
    ani = animation.FuncAnimation(
        fig, 
        func=lambda i: update_traj(i) if i < UPDATE_STEPS else animate_robot(i-UPDATE_STEPS),
        frames=UPDATE_STEPS + N_STEPS, interval=40, blit=True, init_func=init)
    plt.show()
    
    
    # ani.save ('results/maual_trajopt_2d_{}dof_{}_fitting_{}_eps_{}_dif_{}_updates_{}_steps_{}.mp4'.format(
    #     DOF, env_name, fitting_target, Epsilon, dif_weight, UPDATE_STEPS, N_STEPS), fps=FPS)


